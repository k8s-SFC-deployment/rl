vnf_deployment_template = '''
metadata:
    name: {name}
    labels:
        app: {name}
spec:
    replicas: {replicas}
    selector:
        matchLabels:
            app: {name}
    template:
        metadata:
            labels:
                app: {name}
        spec:
            containers:
                - name: {name}
                  image: {image}
                  env:
                    - name: CPU_OPS
                      value: "{CPU_OPS}"
                    - name: CPU_WORKER
                      value: "{CPU_WORKER}"
                    - name: CPU_LIMIT
                      value: "{CPU_LIMIT}"
                    - name: MEM_OPS
                      value: "{MEM_OPS}"
                    - name: MEM_WORKER
                      value: "{MEM_WORKER}"
                    - name: MEM_BYTES
                      value: "{MEM_BYTES}"
                    - name: DIO_OPS
                      value: "{DIO_OPS}"
                    - name: DIO_WORKER
                      value: "{DIO_WORKER}"
                    - name: DIO_BYTES
                      value: "{DIO_BYTES}"
                  ports:
                    - name: http
                      containerPort: 7000
                      protocol: TCP
                  readinessProbe:
                    httpGet:
                        path: /
                        port: 7000
                  resources:
                    limits:
                        cpu: 300m
                        memory: 500Mi
                    requests:
                        cpu: 300m
                        memory: 500Mi
'''

# Example usage:
# name = "vnf-account-0"
# replicas = 1
# image = "euidong/vnf-scc-sfc:prod-0.0.2"
# envs = {
#     "CPU_OPS": "1000",
#     "CPU_WORKER": "1",
#     "CPU_LIMIT": "50",
#     "MEM_OPS": "1000",
#     "MEM_WORKER": "1",
#     "MEM_BYTES": "10000",
#     "DIO_OPS": "1000",
#     "DIO_WORKER": "1",
#     "DIO_BYTES": "10000000",
# }
# deployment = yaml.safe_load(vnf_deployment_template.format(name=name, replicas=replicas, image=image, **envs))
# client.AppsV1Api().create_namespaced_deployment("test-namespace", deployment)

vnf_service_template = '''
metadata:
    name: {name}
    labels:
        app: {name}
spec:
    type: LoadBalancer
    ports:
        - port: 80
          targetPort: 7000
          protocol: TCP
          name: http
    selector:
        app: {name}
'''

# Example usage:
# name = "vnf-account-0"
# service = yaml.safe_load(vnf_service_template.format(name=name))
# client.CoreV1Api().create_namespaced_service("test-namespace", service)
