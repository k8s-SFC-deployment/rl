# Control kubernetes API for VNF container (CNF)
# we have euidong/vnf-scc-sfc:0.0.2 image
# that can relay traffic to each other.
# With this class, we can control (create, delete, scale up, scale down, etc) the container.
# We will make service, and deployment.

import yaml
from typing import Dict, Literal

from src.apis.k8s import K8sApi
from src.env.vnfs.template import vnf_deployment_template, vnf_service_template


class VNF:
    _deploy_template = vnf_deployment_template
    _service_template = vnf_service_template

    def __init__(self, k8sApi: K8sApi, namespace: str, name: str, image: str, envs: Dict[Literal["CPU_OPS", "CPU_WORKER", "CPU_LIMIT", "MEM_OPS", "MEM_WORKER", "MEM_BYTES", "DIO_OPS", "DIO_WORKER", "DIO_BYTES"], str]):
        self.k8sApi = k8sApi

        self.namespace = namespace
        self.name = name
        self.image = image
        self.cur_replicas = 1
        self.envs = envs

    def create(self):
        deployment_body = yaml.safe_load(vnf_deployment_template.format(name=self.name, replicas=self.cur_replicas, image=self.image, **self.envs))
        service_body = yaml.safe_load(vnf_service_template.format(name=self.name))
        self.k8sApi.createDeployment(deployment_body, self.namespace)
        self.k8sApi.createService(service_body, self.namespace)

    def delete(self):
        service_name = self.name
        deploy_name = self.name
        self.k8sApi.deleteService(deploy_name, self.namespace)
        self.k8sApi.deleteDeployment(service_name, self.namespace)

    async def is_ready(self):
        return await self.k8sApi.isAtleastOnePodReadyInDeployment(self.namespace, self.name)

    def scale(self, replicas):
        self.k8sApi.scaleDeployment(self.name, self.namespace, replicas)

    def scaleUp(self, num=1):
        self.cur_replicas += num
        self.scale(self.cur_replicas)