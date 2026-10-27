# Stage 1 — build the React app
FROM node:22-alpine@sha256:fe4f6d74e20370acb04ad889077d14d0f64a9c8ea6a42bc60cddf1e844699914 AS builder
WORKDIR /app

RUN npm install -g pnpm@10.33.0

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .
RUN pnpm build

# Stage 2 — serve via nginx, proxying /api/* to the backend container
FROM nginx:alpine@sha256:a3554c0d0b48177b1413c26e01f695334428020d9df01243fbae3e11cfd17417
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN sed -E -i -e 's@pid[[:space:]]+/(var/)?run/nginx.pid;@pid /tmp/nginx.pid;@' -e '/^user[[:space:]]/d' /etc/nginx/nginx.conf \
    && chmod -R a-w /usr/share/nginx/html /etc/nginx
USER nginx
ENTRYPOINT ["nginx"]
CMD ["-g", "daemon off;"]
EXPOSE 8080
