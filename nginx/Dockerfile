FROM nginx:latest

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our nginx config
COPY conf/nginx.conf /etc/nginx/nginx.conf

# Create directory for SSL certificates (if using SSL)
RUN mkdir -p /etc/nginx/ssl

# Optional: Install any additional tools you might need
RUN apt-get update && apt-get install -y curl vim 