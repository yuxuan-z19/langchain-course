# 基于官方Redis Alpine镜像
FROM redis:7-alpine

# 设置维护者信息
LABEL maintainer="LangChain Course"
LABEL description="Redis for LangChain Memory Storage"

# 安装必要的工具
RUN apk add --no-cache \
    bash \
    curl \
    && rm -rf /var/cache/apk/*

# 创建Redis配置目录
RUN mkdir -p /usr/local/etc/redis

# 复制自定义Redis配置文件
COPY redis.conf /usr/local/etc/redis/redis.conf

# 设置Redis数据目录权限
RUN chown -R redis:redis /data /usr/local/etc/redis

# 创建健康检查脚本
RUN echo '#!/bin/bash\nredis-cli ping' > /usr/local/bin/health-check.sh \
    && chmod +x /usr/local/bin/health-check.sh

# 暴露Redis端口
EXPOSE 6379

# 设置工作目录
WORKDIR /data

# 切换到redis用户
USER redis

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# 启动Redis服务
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]