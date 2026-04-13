# Unified Tree 接入 DeepSeek V4 L2 HiCache 设计

> **范围**：本文档只讨论 L2 HiCache，即 `UnifiedRadixCache` 管理的 GPU ↔ Host 这一层。**不讨论** L3 storage/prefetch、SWA offload、decode-ready restore。

## 概述

Unified Radix Tree 通过将 full-token 逻辑索引作为唯一的树级缓存身份来支持 DeepSeek V4。在 L1 上，它保留逻辑 full indices 以及仅存于 device 的 SWA 状态；在 L2 上，它只存储一个逻辑 full host span，不维护真正的 full-KV host buffer。当前设计中，HiCache offloading 只覆盖压缩 KV 路径，即 c4、c4\_indexer 和 c128，不 offload SWA。这些压缩池索引在 HiCache backup 和 load-back 时直接从逻辑 full span 派生，因此树不需要将它们作为独立的树状态存储。压缩 KV 加载回 device 后，运行时通过 replay 剩余的 SWA 部分前缀来重建 SWA 滑动窗口和临时压缩状态，然后才继续 decode。

为了支持这条路径，HybridCacheAssembler 需要构建一个 V4 专用的压缩缓存栈，取代当前的 KV-anchor 栈。在 HiCache 侧，还需要为 c4、c128 和 indexer 创建真正的 host pool，因为只有这些才是 device 与 host 之间实际传输的物理缓存对象，而 full host span 保持纯逻辑。

因此，L2 host 命中是 **compressed-ready，而非 decode-ready**。压缩 KV 加载回 device 后，运行时 replay 剩余的尾部 token 以重建 SWA 窗口和临时压缩状态，然后才继续 decode。


## 1. 背景与约束

### 1.1 V4 KV 结构

V4 的 KV 池并非 `Full KV + SWA KV`，而是：

| 池 | 说明 |
|----|------|
| `FULL` | 逻辑 token 索引空间（无真实 KV 数据） |
| `SWA` | 滑动窗口 device 缓存（未压缩） |
| `C4` | 压缩 KV（压缩比 = 4） |
| `C128` | 压缩 KV（压缩比 = 128） |
| `C4 Indexer` | C4 的 sidecar |
| `compress_state` | 绑定在 SWA loc 上的 ring buffer |

对 L2 HiCache 而言，值得 offload 的只有：**C4 main**、**C4 indexer**、**C128 main**。

不 offload：Full KV、SWA KV、compress\_state。


### 1.2 Unified Tree 当前假设

`UnifiedRadixCache` 用 `FULL` 同时作为树骨架和 host-backed 判定依据：

- `node.backuped := FULL.host_value is not None`
- `node.evicted := FULL.value is None`
- `child.evicted and not child.backuped` → dead node
- `host_hit_length` 从 `FULL.host_value` 链推导

这套状态机保持不变。在 V4 下，`FULL.host_value` 变成**逻辑 host full span**而非真实的 host KV 索引。它来自一个只做索引分配的 `LogicalHostPool`：

```python
FULL.host_value = logical_host_pool.alloc(node_len)
```

它不对应任何真实的 Full KV host buffer，唯一目的是保持 `len()`、slice、clone、split 可用，使得树状态机不需要修改。


### 1.3 Controller 不是瓶颈

`HybridCacheController` 已经支持 `kv_indices.numel() == 0` 和仅 `extra_pools` 的传输。主要改动在树侧和索引 resolver。


## 2. 组件架构

### 2.1 Tree Components

V4 unified tree 使用三个组件：

```python
tree_components = (ComponentType.FULL, ComponentType.SWA, ComponentType.DSV4_COMPRESSED)
```

```python
class ComponentType(int, Enum):
    FULL = 0
    SWA = 1
    MAMBA = 2
    DSV4_COMPRESSED = 3

_NUM_COMPONENT_TYPES = 4
```

| 组件 | Device 状态 | Host 状态 | HiCache 行为 |
|------|-----------|---------|-------------|
| **FULL** | 逻辑 full indices (`value`) | 逻辑占位符 (`host_value`) | backup/load 的锚点（无真实 DMA） |
| **SWA** | 滑动窗口 device 缓存 | 无 | **不 offload** |
| **DSV4\_COMPRESSED** | 节点上无状态 | 节点上无状态 | 从 FULL 派生 C4/C128 索引，生成 PoolTransfer |


### 2.2 DSV4\_COMPRESSED 组件

该组件**不携带任何节点级状态**。它仅通过 HiCache hook 参与，在传输时从 full 逻辑 span 实时派生压缩索引。

```python
class DeepSeekV4CompressedComponent(TreeComponent):
    component_type = ComponentType.DSV4_COMPRESSED

    def __init__(self, cache, params):
        super().__init__(cache, params)
        self.c4_ratio = 4
        self.c128_ratio = 128
```

关键行为：

- `allocate_for_node`：无操作
- `free_node`：无操作
- `redistribute_on_node_split`：无操作（节点上无数据）
- `evict_component`：无操作（无数据可释放）
- `create_match_validator`：始终返回 True（压缩状态跟随 full）
- `build_hicache_transfers`：从 `cd[FULL].value` 或 `cd[FULL].host_value` 派生索引
- `commit_hicache_transfer`：load-back 后更新 device 侧映射表


### 2.3 SWA 组件

SWA 组件保持现有逻辑，不参与 HiCache hook：

- 管理 SWA 生命周期：tombstone、eviction、滑动窗口 match validation
- `build_hicache_transfers`：返回 `None`（默认行为）
- `drive_host_eviction`：无操作
- 启动时必须存在，用于 match validation 的正确性


## 3. Host 池结构

### 3.1 Device 侧分析

Device 侧 C4 和 C128 使用**同一个类** `DeepSeekV4SingleKVPool`，区别仅在构造参数（`size`、`page_size`、`layer_num`）。两者共用 `kv_cache_total_dim = 584` 字节/token，buffer 布局完全相同：

```python
# 每层一个 2D tensor，paged 布局，576 字节对齐
kv_buffer[layer_id] = torch.zeros(num_pages, bytes_per_page_padded, dtype=uint8)
# bytes_per_page_padded = ceil_div(page_size * kv_cache_total_dim, 576) * 576
```

C4 Indexer 使用**不同的类** `DeepSeekV4IndexerPool`，布局模式相同但 dim 计算不同且无 576 对齐：

```python
# 每层一个 2D tensor，无 576 对齐
index_k_with_scale_buffer[layer_id] = torch.zeros(num_pages, page_bytes, dtype=uint8)
# page_bytes = page_size * index_head_dim + page_size * (index_head_dim // 128) * 4
```

三者布局模式一致（`[num_pages, bytes_per_page]` per layer, uint8），但 C4/C128 与 Indexer 在 dim 计算和对齐方式上有本质区别。

### 3.2 Host 侧两个类

遵循 device 侧的类划分，host 侧用**两个类**：

```
DeepSeekV4CompressedKVHostPool   → C4 host（实例）、C128 host（实例）
DeepSeekV4IndexerHostPool        → C4 Indexer host（实例）
```

C4/C128 不需要独立实现，用同一个类的不同实例。Indexer 独立一个类，不混杂。

### 3.3 池拓扑

```
LogicalHostPool（现有 HostKVCache，或基于 arange 的逻辑池）
├── size = full_host_size
├── free_list: alloc(N) / free(indices)
└── V4 下可以是纯逻辑的（无真实 KV buffer）

C4 Host Pool（DeepSeekV4CompressedKVHostPool 实例）
├── num_host_pages = full_host_size // full_page_size  （与 full page 1:1 对应）
├── 无独立 allocator —— 索引为 page index，从 full 派生
└── kv_buffer: list[layer_num], 每层 [num_host_pages, bytes_per_page_padded]

C128 Host Pool（DeepSeekV4CompressedKVHostPool 实例）
├── num_host_pages = full_host_size // full_page_size  （与 full page 1:1 对应）
├── 无独立 allocator
└── kv_buffer: list[layer_num], 每层 [num_host_pages, bytes_per_page_padded]

C4 Indexer Host Pool（DeepSeekV4IndexerHostPool 实例）
├── num_host_pages = full_host_size // full_page_size
├── 无独立 allocator
└── kv_buffer: list[layer_num], 每层 [num_host_pages, indexer_page_bytes]
```

### 3.4 索引派生

Device 侧的 buffer 是 paged 布局，所以 HiCache DMA 以 page 为粒度传输。索引需要从 full token index 派生为 page index。

以 `full_page_size = 256` 为例：

```
C4:   c4_page_size   = full_page_size // 4   = 64
      c4_page_index  = (full_token // 4) // 64 = full_token // 256 = full_token // full_page_size

C128: c128_page_size = full_page_size // 128 = 2
      c128_page_index = (full_token // 128) // 2 = full_token // 256 = full_token // full_page_size
```

本质原因：`compress_ratio × compressed_page_size = full_page_size`，所以 `(full // ratio) // (full_ps // ratio)` 恒等于 `full // full_ps`。一个 full page 的 256 个 token 刚好被压缩成一个 C4 page（64 token）和一个 C128 page（2 token），是 1:1 的对应关系。

因此三个 pool 共用同一个派生函数：

```python
page_derive = lambda full_indices: torch.unique(full_indices // full_page_size)
```

### 3.5 V4 Host Pool 实现

`HostPoolGroup` 在收到 `pool_transfers` 时，根据 `PoolEntry.name` 查找 entry，调用 `host_pool` 执行 DMA。

#### 3.5.1 DeepSeekV4CompressedKVHostPool（C4/C128 共用）

Host buffer 镜像 device 的 paged 布局：

```python
class DeepSeekV4CompressedKVHostPool:
    """C4 / C128 compressed KV 的 host pool。
    同一个类的不同实例分别服务 C4 和 C128。"""

    def __init__(self, device_pool: DeepSeekV4SingleKVPool, num_host_pages: int):
        self.layer_num = device_pool.layer_num
        self.bytes_per_page_padded = device_pool.bytes_per_page_padded
        self.dtype = torch.uint8

        # 每层一个 2D tensor，与 device 布局一致，pinned memory
        self.kv_buffer = [
            torch.zeros(num_host_pages, self.bytes_per_page_padded,
                        dtype=self.dtype, device="cpu")
            for _ in range(self.layer_num)
        ]
        # cudaHostRegister
        for buf in self.kv_buffer:
            torch.cuda.cudart().cudaHostRegister(
                buf.data_ptr(), buf.numel() * buf.element_size(), 0)

        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64, device=device_pool.device)

        # 无 alloc / free / free_slots —— 索引从 full host 派生
```

DMA 使用标准 MLA 系列 transfer 内核，`item_size = bytes_per_page_padded`（整页传输）：

```python
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend=None
    ) -> None:
        """D→H: page-level transfer, all layers."""
        transfer_kv_all_layer_mla(
            src_layers=device_pool.data_ptrs,   # device per-layer base ptrs
            dst_layers=self.data_ptrs,           # host per-layer base ptrs
            src_indices=device_indices,           # page indices
            dst_indices=host_indices,             # page indices
            item_size=self.bytes_per_page_padded, # 整页字节数
            num_layers=self.layer_num,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend=None
    ) -> None:
        """H→D: page-level transfer, one layer."""
        transfer_kv_per_layer_mla(
            src=self.kv_buffer[layer_id],
            dst=device_pool.kv_buffer[layer_id],
            src_indices=host_indices,
            dst_indices=device_indices,
            item_size=self.bytes_per_page_padded,
        )
```

#### 3.5.2 DeepSeekV4IndexerHostPool（C4 Indexer 专用）

与 C4/C128 host pool 结构相同，但 bytes_per_page 不同且无 576 对齐：

```python
class DeepSeekV4IndexerHostPool:
    """C4 Indexer 的 host pool。"""

    def __init__(self, device_pool: DeepSeekV4IndexerPool, num_host_pages: int):
        self.layer_num = device_pool.layer_num
        # Indexer 的 page_bytes 计算方式不同，无 576 对齐
        num_scales_per_token = device_pool.index_head_dim // device_pool.quant_block_size
        self.page_bytes = (
            device_pool.page_size * device_pool.index_head_dim
            + device_pool.page_size * num_scales_per_token * 4
        )
        self.dtype = torch.uint8

        # 每层一个 2D tensor，pinned memory
        self.kv_buffer = [
            torch.zeros(num_host_pages, self.page_bytes,
                        dtype=self.dtype, device="cpu")
            for _ in range(self.layer_num)
        ]
        for buf in self.kv_buffer:
            torch.cuda.cudart().cudaHostRegister(
                buf.data_ptr(), buf.numel() * buf.element_size(), 0)

        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64, device=device_pool.device)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend=None
    ) -> None:
        transfer_kv_all_layer_mla(
            src_layers=device_pool_data_ptrs,  # 需要从 device_pool 构建
            dst_layers=self.data_ptrs,
            src_indices=device_indices,
            dst_indices=host_indices,
            item_size=self.page_bytes,
            num_layers=self.layer_num,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend=None
    ) -> None:
        transfer_kv_per_layer_mla(
            src=self.kv_buffer[layer_id],
            dst=device_pool.index_k_with_scale_buffer[layer_id],  # 注意 buffer 名不同
            src_indices=host_indices,
            dst_indices=device_indices,
            item_size=self.page_bytes,
        )
```

#### 3.5.3 两类对比

| | C4/C128 Host Pool | C4 Indexer Host Pool |
|--|-------------------|---------------------|
| Host 类 | `DeepSeekV4CompressedKVHostPool` | `DeepSeekV4IndexerHostPool` |
| Device 池类 | `DeepSeekV4SingleKVPool` | `DeepSeekV4IndexerPool` |
| Device buffer 名 | `kv_buffer` | `index_k_with_scale_buffer` |
| bytes_per_page | `ceil_div(ps * 584, 576) * 576` | `ps * index_head_dim + ps * scales * 4` |
| 576 对齐 | 有 | 无 |
| DMA 内核 | `transfer_kv_*_mla` | `transfer_kv_*_mla` |
| 索引语义 | page index | page index（相同） |

注：`DeepSeekV4IndexerPool` 没有 `data_ptrs` 属性，需要在 host pool 初始化时从 `index_k_with_scale_buffer` 构建。

`PoolEntry` 需要 `layer_mapper`，用于 V4 注意力层到 pool buffer 层的映射（V4 不是所有层都有 C4/C128）。

### 3.6 Device 侧 layer-wise loading 同步

HiCache load-back 支持 per-layer 传输：host → device 逐层 DMA，前端 forward 在访问某层 buffer 时需要等待该层传输完成。现有池通过 `layer_transfer_counter.wait_until(layer_id - start_layer)` 实现同步。

**已有 `wait_until` 的池（参考实现）：**

- `MHATokenToKVPool.get_key_buffer()` / `get_value_buffer()`
- `MLATokenToKVPool.get_key_buffer()` / `get_value_buffer()`
- `NSATokenToKVPool.get_index_k_with_scale_buffer()`

**V4 当前缺失 `wait_until`：**

- `DeepSeekV4SingleKVPool.get_key_buffer()` — 直接返回 buffer，无等待
- `DeepSeekV4IndexerPool.get_index_k_with_scale_buffer()` — 直接返回，无等待

**正确做法：在容器 `DeepSeekV4TokenToKVPool` 的访问入口加 `wait_until`，而非子池。**

原因：子池只知道自己的 `compress_layer_id`（如 C4 有 46 层），不知道全局 `layer_id`；而 `LayerDoneCounter` 按全局 layer 顺序推进。容器持有全局 `layer_id`，是正确的同步点。

需要加 `wait_until` 的入口：

```python
# DeepSeekV4TokenToKVPool 中：
def get_swa_key_buffer(self, layer_id):
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
    return self.swa_kv_pool.get_key_buffer(layer_id)

def get_extra_key_buffer(self, layer_id):
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
    _, compress_layer_id, compress_kv_pool = self.layer_mapping[layer_id]
    return compress_kv_pool.get_key_buffer(compress_layer_id)

def get_index_k_with_scale_buffer(self, layer_id):
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
    compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
    return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(compress_layer_id)
```

此外，`register_layer_transfer_counter` 由 `HiCacheController` 调用，注册在 `DeepSeekV4TokenToKVPool`（容器）上即可，不需要传播到子池。

### 3.7 池名称

```python
class PoolName(str, Enum):
    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    INDEXER = "indexer"
    C4 = "c4"              # 新增
    C128 = "c128"           # 新增
    C4_INDEXER = "c4_indexer"  # 新增
```


## 4. PoolEntry 与 Controller 改动

### 4.1 PoolEntry 扩展

```python
@dataclass
class PoolEntry:
    name: PoolName
    host_pool: Any
    device_pool: Any
    layer_mapper: Optional[Callable] = None
    is_primary_index_anchor: bool = False
    host_evict_fn: Optional[Callable] = None
    device_evict_fn: Optional[Callable] = None
    share_indices_with_anchor: bool = False
    # --- 新增 ---
    derive_indices_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
```

`derive_indices_fn` 是一个通用的索引派生回调：接收 full indices，返回 compressed indices。由 Assembler 在注册时设置，resolver 不需要了解 V4 特有逻辑。

注册示例（索引为 page-level）：

```python
# 所有压缩池的 page index 等于 full page index
page_derive = lambda full: torch.unique(full // full_page_size)

PoolEntry(name=PoolName.C4,         host_pool=c4_host,         device_pool=c4_device,
          derive_indices_fn=page_derive)
PoolEntry(name=PoolName.C4_INDEXER,  host_pool=c4_indexer_host, device_pool=c4_indexer_device,
          derive_indices_fn=page_derive)   # 同一个 page_derive
PoolEntry(name=PoolName.C128,        host_pool=c128_host,       device_pool=c128_device,
          derive_indices_fn=page_derive)   # 同一个 page_derive
```

三个 pool 共用同一个 `page_derive`，因为一个 full page 恰好对应一个 C4 page、一个 C128 page、一个 Indexer page。


### 4.2 `_resolve_pool_transfers_allocation` —— 新增派生模式

三种解析模式，按顺序检查：

```python
def _resolve_pool_transfers_allocation(self, extra_pools, alloc_host,
                                        kv_device_indices, kv_host_indices):
    for pool in extra_pools:
        entry = self.mem_pool_host.entry_map.get(pool.name)

        # 模式 1：与 anchor 共享索引（已有）
        if entry.share_indices_with_anchor:
            pool.device_indices = kv_device_indices
            pool.host_indices = kv_host_indices
            continue

        # 模式 2：通过回调从 full indices 派生（新增）
        if entry.derive_indices_fn:
            if alloc_host:   # BACKUP 方向
                pool.host_indices = entry.derive_indices_fn(kv_host_indices)
            else:            # LOAD_BACK 方向
                pool.device_indices = entry.derive_indices_fn(kv_device_indices)
            continue

        # 模式 3：独立分配（已有）
        ...
```

**为什么派生逻辑在 resolver 而非 V4Component 中？**

时序原因。BACKUP 时 V4Component 的 `build_hicache_transfers` 先于 `controller.write()` 运行，此时 `full_host_indices` 尚未分配，V4Component 只能填 `device_indices`。LOAD_BACK 同理，`full_device_indices` 尚未分配。Resolver 在 alloc 之后运行，自然拥有刚分配的 full indices，是填补缺失一侧的正确时机。

`derive_indices_fn` 作为回调而非硬编码字段，使得 resolver 本身保持通用 —— 不包含任何 V4 特有逻辑。

派生模式不需要 rollback 逻辑 —— 没有分配就没有回滚。


## 5. 写入路径：`write_backup`

### 5.1 目标

当 device 节点需要备份时：

- **不**传输 Full KV 到 host（没有真实的 Full host buffer）
- 只通过压缩 host 池传输 C4 / C4 Indexer / C128
- 成功后向 `FULL.host_value` 写入逻辑占位符

### 5.2 流程

```
1. V4Component.build_hicache_transfers(BACKUP_HOST):
     full_device = cd[FULL].value
     page_dev = unique(full_device // full_page_size)
     → [PoolTransfer(C4,         device_indices=page_dev),
        PoolTransfer(C4_INDEXER, device_indices=page_dev),
        PoolTransfer(C128,       device_indices=page_dev)]
     # host_indices 留空，由 resolver 派生
     # 三个 pool 共用同一个 page_dev

2. controller.write():
     host_indices = LogicalHostPool.alloc(len(device_value))
     _resolve: 对 derive_indices_fn 的 C4/C128/Indexer entry：
         pool.host_indices = page_derive(host_indices)  # = unique(host_indices // full_page_size)
     DMA: 对每个压缩池执行 device → host（page-level transfer）

3. FullComponent.commit(BACKUP_HOST):
     cd[FULL].host_value = host_indices   # 逻辑占位符

4. V4Component.commit(BACKUP_HOST):
     无操作（host 索引可从 FULL.host_value 重新派生，无需存储）
```

### 5.3 树级改动

当前 `write_backup()` 按 Full host 池预检查 `host_avail < kv_tokens`。V4 下：

- Full host 池是逻辑的 —— 其"容量"就是 free-list 大小
- 真实的 host 内存压力来自 C4/C128/Indexer host 池
- 预回收应检查压缩池的可用性，而非 Full KV 的可用性
- 或者由 Controller 的 `host_evict_fn` 在每个 PoolEntry 上惰性处理回收


## 6. Device 降级：`_evict_to_host`

树级语义保持不变：

| 组件 | device eviction 行为 |
|------|---------------------|
| FULL | `FULL.value = None`（device 索引释放） |
| SWA | `SWA.value = None`（SWA device 槽位释放） |
| DSV4\_COMPRESSED | 无操作（节点上无数据） |

降级后状态：

- `FULL.value = None`，`FULL.host_value != None` → `node.evicted = True`，`node.backuped = True`
- 压缩 KV 数据存在于 C4/C128 host 池中，可通过 `unique(FULL.host_value // full_page_size)` 寻址
- SWA 数据已丢失（不 offload）


## 7. 匹配路径：`match_prefix`

### 7.1 目标语义

V4 不引入新的 match hook，也不把树改成显式的 two-frontier 框架。`match_prefix` 仍然只做一次 radix path 遍历，并继续使用现有的 `create_match_validator()`。

区别在于：对 V4 而言，这次遍历需要同时记住两类结果：

- `best_node`：最深的 host-recoverable 节点
- `best_device_node`：最深的 device-ready 节点

这两个结果都在 `_match_prefix_helper()` 的同一次遍历中得到。`match_post` 只负责组装 `MatchResult`，而不再尝试从 host frontier 反推 device frontier。

### 7.2 FULL Validator

不变：`FULL.value is not None or node.backuped`。

### 7.3 SWA Validator

V4 下，SWA validator 不再阻塞 host-backed 的匹配：

- `SWA.value != None` → 标准滑动窗口校验
- `node.backuped` → 返回 `True`（host-backed 节点，SWA 将在 replay 中重建）
- 其他情况 → `False`

这意味着 `match_prefix` 可以继续走到更深的 host 节点，即使这些节点已经没有 live SWA。

### 7.4 DSV4\_COMPRESSED Validator

始终返回 `True`。压缩数据跟随 FULL：节点进入 backed-up 状态时，C4 / C4 Indexer / C128 也已经完成 host backup。

### 7.5 `_match_prefix_helper()` 需要同时记录 host 和 device frontier

当前 `UnifiedRadixCache` 里，`best_value_len` 会被用来构造 `device_indices`，而 `last_device_node` 会被调度器存进 `req.last_node`。这两者必须保持对齐：

- `prefix_indices must always be aligned with last_node`
- `req.prefix_indices` 和 `req.last_node` 会一起参与后续 lock / insert / 调度

因此，V4 不能采用“先找最深 host node，再在 `match_post` 中回溯 device node”的方案。原因是 SWA 的 device-ready 语义不是节点局部属性，而是带路径状态的 validator；仅靠 `FULL.value != None && SWA.value != None` 不能可靠恢复出真实的 device frontier。

正确做法是在 `_match_prefix_helper()` 中单次遍历时同时维护：

- `best_node`
- `best_value_len`
- `best_device_node`
- `best_device_value_len`

更新规则如下：

- 只要当前节点通过现有 validators，就更新 `best_node`
- 只有当当前节点同时仍是 live device node 时，才更新 `best_device_node`
- `best_value_len` 对应 host frontier
- `best_device_value_len` 对应 device frontier

最后：

- `device_indices = torch.cat(value_chunks[:best_device_value_len])`
- `last_device_node = best_device_node`
- `last_host_node = best_node`（或其最近的 `backuped` ancestor）

这样：

- host frontier 可以比 device frontier 更深
- `device_indices` 与 `last_device_node` 仍然严格对齐
- scheduler 不需要理解新的语义分叉

### 7.6 Host Hit Length

L2 host 命中是 compressed-ready，不是 decode-ready。对 V4 来说，应先计算 host 相比 device 多出来的那段前缀，再为 replay 预留一个 page：

```python
raw_host_hit_length = prefix_len(last_host_node) - prefix_len(last_device_node)
host_hit_length = max(0, raw_host_hit_length - replay_window)
```

对 V4 paged 模式（`sliding_window_size=128`，`page_size=256`）：`replay_window = 256`。

Scheduler 将 `host_hit_length` 视为可恢复前缀；剩余的尾部 token 走正常 extend，自然重建 SWA + compress_state。


## 8. 加载路径：`load_back`

### 8.1 目标

从 host 恢复 compressed-ready 前缀到 device：

- 只加载 `host_hit_length` 覆盖的压缩 KV
- **不**恢复 SWA 和 compress\_state
- 尾部窗口由 scheduler 通过 replay 恢复

### 8.2 流程

```
1. FullComponent.build_hicache_transfers(LOAD_BACK):
     沿 evicted chain 向上收集 FULL.host_value（逻辑占位符）
     → PoolTransfer(KV, host_indices=cat(placeholders), device_indices=None,
                    nodes_to_load=[...])
     # 如果使用纯逻辑模式，host_indices.numel() 可能为 0

2. V4Component.build_hicache_transfers(LOAD_BACK):
     full_host = cd[FULL].host_value
     page_host = unique(full_host // full_page_size)
     → [PoolTransfer(C4,         host_indices=page_host),
        PoolTransfer(C4_INDEXER, host_indices=page_host),
        PoolTransfer(C128,       host_indices=page_host)]
     # device_indices 留空，由 resolver 派生
     # 三个 pool 共用同一个 page_host

3. controller.load():
     full_device = allocator.alloc(len(host_indices))  # 逻辑 full 索引
     _resolve: 对 derive_indices_fn 的 C4/C128/Indexer entry：
         pool.device_indices = page_derive(full_device)  # = unique(full_device // full_page_size)
     DMA: 对每个压缩池执行 host → device（page-level transfer）

4. FullComponent.commit(LOAD_BACK):
     在每个加载的节点上恢复 cd[FULL].value = full_device

5. V4Component.commit(LOAD_BACK):
     full_device = cd[FULL].value  # 由 FullComponent 在上一步设置
     按需更新 device 侧映射表
```

### 8.3 为什么这条路径可行

- Controller 已支持 `kv_indices.numel() == 0`
- C4/C128 的 device 索引不需要 controller 分配 —— 由 resolver 从 full device 索引派生（page-level）
- Controller 只负责执行 DMA 传输


## 9. 节点生命周期总表

| 操作 | FULL | SWA | DSV4\_COMPRESSED |
|------|------|-----|-----------------|
| allocate | 分配逻辑索引 | 分配 SWA device 槽位 | 无操作 |
| free（device evict） | 释放逻辑索引 | 释放 SWA device 槽位 | 无操作 |
| backup（D→H） | 存储逻辑占位符 | 不 offload | 派生 C4/C128 索引，DMA |
| load\_back（H→D） | 恢复逻辑 value | 不恢复（通过 replay） | 派生索引，DMA，更新映射 |
| split / redistribute | slice value + host\_value | slice SWA value | 无操作（从 FULL 派生） |


## 10. Assembler 改动

### 10.1 新增 V4 Stack Builder

`hybrid_pool_assembler.py` 在 `attach_hybrid_pool_to_unified_cache` 中需要新增分支：

```python
if isinstance(kvcache, DeepSeekV4TokenToKVPool):
    build_deepseekv4_compressed_stack(kvcache, host_size, ...)
```

`build_deepseekv4_compressed_stack` 职责：

1. 创建 `LogicalHostPool`（size = `full_host_size`，纯逻辑，无真实 KV buffer）
2. 创建 `DeepSeekV4CompressedKVHostPool`（C4 实例，num_host_pages = full_host_size // full_page_size）
3. 创建 `DeepSeekV4CompressedKVHostPool`（C128 实例，同 num_host_pages，不同 device_pool）
4. 创建 `DeepSeekV4IndexerHostPool`（C4 Indexer 实例，同 num_host_pages）
5. 注册到 `HostPoolGroup`，使用 `PoolEntry(derive_indices_fn=page_derive)`


## 11. Replay 语义

### 11.1 为什么需要 replay

L2 load-back 只恢复压缩 KV（C4 / C128 / Indexer）。SWA 和 compress\_state **不会**被恢复。

### 11.2 Replay 如何发生

由 `host_hit_length` 驱动：

- Scheduler 将 `host_hit_length` 视为可恢复前缀
- `req.extend_input_len` 自然保留最后 `replay_window` 个 token
- 这些 token 走正常 extend 流程，顺便重建 SWA + compress\_state
- Scheduler 中不需要特殊的 "tail replay mode"


## 12. 分阶段落地

### Phase 1：语义打底

- 在 `tree_component.py` 中新增 `ComponentType.DSV4_COMPRESSED`
- 在 `hicache_storage.py` 中新增 `PoolName.C4 / C128 / C4_INDEXER`
- 将 `FULL.host_value` 切换到逻辑 host full span 语义
- 在 `PoolEntry` 上新增 `derive_indices_fn` 回调（page-level 派生）
- 实现 `DeepSeekV4CompressedComponent`（节点生命周期无操作，仅 HiCache hook）
- 更新 SWA validator 使其不阻塞 host-backed 匹配
- 修改 `_match_prefix_helper()`：在同一次遍历里同时记录 host frontier 和 device frontier
- 修改 `match_post()`：直接消费 helper 返回的 `best_device_node / best_device_value_len`

### Phase 2：L2 写入路径

- 在 V4 组件中实现 `build_hicache_transfers(BACKUP_HOST)`
- 在 `_resolve_pool_transfers_allocation` 中新增派生模式
- 修改 `write_backup()` 为 aux-only 备份（无 Full KV DMA）
- 在 assembler 中实现 `build_deepseekv4_compressed_stack`

### Phase 3：L2 加载路径

- 在 V4 组件中实现 `build_hicache_transfers(LOAD_BACK)`
- 实现 `commit_hicache_transfer(LOAD_BACK)` 含映射表更新
- 将 `host_hit_length` 切换到 compressed-ready 语义
- 修改 `load_back()` 为按 `host_hit_length` 截断恢复，并将最后一个 `256-token` page 留给 replay

### Phase 4：验证

- 单请求 write\_backup / load\_back 正确性
- 节点 split 正确性（FULL 占位符正确拆分，压缩数据可重新派生）
- host leaf / device leaf / tombstone 状态机
- tail replay 正确性（load-back 后 SWA + compress\_state 重建）
- 混合 C4/C128 层配置正确性
