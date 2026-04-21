FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json ./
# --legacy-peer-deps: some of our deps (recharts, lucide-react, etc.) still
# declare react@^18 in peerDependencies while we're on react@19. Dropping this
# flag once the ecosystem catches up is safe.
RUN npm install --legacy-peer-deps --omit=optional

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
# No public/ dir in this app — nothing to copy.

EXPOSE 3000
CMD ["node", "server.js"]
