from kubernetes.client.rest import ApiException
from kubernetes import client, config, watch
from collections import defaultdict

def initApi(config_file):
    return K8sApi(config_file=config_file)


class K8sApi:
    api: client.CoreV1Api
    api_apps: client.AppsV1Api

    def __init__(self, config_file):
        config.load_kube_config(config_file=config_file)
        self.api = client.CoreV1Api()
        self.api_apps = client.AppsV1Api()

    def getNodes(self):
        node_list = self.api.list_node(watch=False)
        return node_list.items

    def getServices(self, namespace=None):
        if namespace:
            service_list = self.api.list_namespaced_service(namespace, watch=False)
        else:
            service_list = self.api.list_service_for_all_namespaces(watch=False)
        return service_list.items

    def getPods(self, namespace=None):
        if namespace:
            pod_list = self.api.list_namespaced_pod(namespace, watch=False)
        else:
            pod_list = self.api.list_pod_for_all_namespaces(watch=False)
        return pod_list.items

    def getPodsByNode(self, namespace=None):
        pods = self.getPods(namespace)
        pods_by_node = defaultdict(list)
        for pod in pods:
            if pod.spec.node_name is None:
                pods_by_node["Unschedulable"].append(pod.metadata.name)
            else:
                pods_by_node[pod.spec.node_name].append(pod.metadata.name)
        return pods_by_node
    
    def deleteAll(self, namespace):
        self.api.delete_collection_namespaced_service(namespace=namespace)
        self.api_apps.delete_collection_namespaced_deployment(namespace=namespace)


    def createNamespace(self, namespace):
        try:
            return self.api.read_namespace(namespace)
        except ApiException as e:
            if e.status == 404:
                namespace = {
                    "metadata": {
                        "name": namespace,
                        "labels": {
                            "istio-injection": "enabled",
                        },
                    },
                }
                return self.api.create_namespace(namespace)
            else:
                raise e
    
    def deleteNamespace(self, namespace):
        try:
            # 네임스페이스 삭제 요청
            self.api.delete_namespace(name=namespace)

            # 완전히 삭제될 때까지 대기
            w = watch.Watch()
            for event in w.stream(self.api.list_namespace, timeout_seconds=600):
                if event['object'].metadata.name == namespace and event['type'] == 'DELETED':
                    break
        except ApiException as e:
            if e.status == 404:
                print(f"Namespace '{namespace}' not found.")
            else:
                raise e

    def createDeployment(self, body, namespace):
        return self.api_apps.create_namespaced_deployment(namespace, body)
        

    def createService(self, body, namespace):
        return self.api.create_namespaced_service(namespace, body)
        
    def deleteService(self, name, namespace):
        try:
            return self.api.delete_namespaced_service(namespace, name)
        except ApiException as e:
            if e.status == 404:
                return None
            else:
                raise e

    def deleteDeployment(self, name, namespace):
        try:
            return self.api_apps.delete_namespaced_deployment(namespace, name)
        except ApiException as e:
            if e.status == 404:
                return None
            else:
                raise e

    def scaleDeployment(self, name, namespace, replicas):
        deployment = self.api_apps.read_namespaced_deployment(name, namespace)
        deployment.spec.replicas = replicas
        return self.api_apps.patch_namespaced_deployment(name=name, namespace=namespace, body=deployment)

    def isAtleastOnePodReadyInDeployment(self, name, namespace):
        # 쿠버네티스 클라이언트 설정
        try:
            w = watch.Watch()
            for event in w.stream(self.api_apps.list_namespaced_deployment, namespace=namespace, timeout_seconds=600):
                deployment = event['object']
                if deployment.metadata.name == name:
                    if deployment.status.ready_replicas and deployment.status.ready_replicas > 0:
                        break
        except ApiException as e:
            print(f"An error occurred: {e}")


if __name__ == '__main__':
    k8sApi = initApi(config_file='D:\projects\GRLScheduler\kube_config.yaml')
    nodes = k8sApi.getNodes()
    pods = k8sApi.getPods()
    print(
        f"Number of nodes: {len(nodes)}\nNumber of pods: {len(pods)}")
    pods_by_node = k8sApi.getPodsByNode()
    for node, pods in pods_by_node.items():
        print(f"Node: {node}\nPods: {pods}")
