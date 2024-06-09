from src.apis.k8s import initApi as initK8sApi
from src.apis.prom import initApi as initPromApi

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List


class Collector:
    def __init__(self, k8sApi, promApi, namespace, duration: int = 60):
        self.k8sApi = k8sApi
        self.promApi = promApi
        self.namespace = namespace
        self.duration = f"{duration}s"

    def getNodes(self):
        """
        Return:
        - nodeVertexDf
            - type: pd.DataFrame
            - columns: (id, hostname, ip, cpu_cap, mem_cap, cpu_util, mem_util, receive_bytes, transmit_bytes)
        - nodeEdgeDf
            - type: pd.DataFrame
            - columns: (id, src, dst, receive_bytes, transmit_bytes, latency_microseconds)
        """
        # <-- get node vertex information -->
        k8sNodes = self.k8sApi.getNodes()
        k8sNodeInfos = []
        for k8sNode in k8sNodes:
            k8sNodeInfo = defaultdict()
            for address in k8sNode.status.addresses:
                if address.type == 'InternalIP':
                    k8sNodeInfo["ip"] = address.address
                if address.type == 'Hostname':
                    k8sNodeInfo["hostname"] = address.address
            k8sNodeInfo["cpu_cap"] = int(k8sNode.status.capacity['cpu'])
            k8sNodeInfo["mem_cap"] = int(
                k8sNode.status.capacity['memory'][:-2])  # (cut, kb) - KiB
            k8sNodeInfos.append(k8sNodeInfo)
        # (id, ip, hostname, cpu_cap, mem_cap)
        k8sNodeDf = pd.DataFrame(k8sNodeInfos)

        promNodeCpuUtils = self.promApi.getCpuUtilizationByNode(self.duration)
        promNodeCpuDict = map(lambda b: {"hostname": b["metric"]["node"], "cpu_util": float(b["value"][1])}, promNodeCpuUtils)
        # (id, hostname, cpu_util)
        promNodeCpuDf = pd.DataFrame(promNodeCpuDict)
        
        promNodeMemUtils = self.promApi.getMemoryUtilizationByNode()
        promNodeMemDict = map(lambda b: {"hostname": b["metric"]["node"], "mem_util": float(b["value"][1])}, promNodeMemUtils)
        # (id, hostname, mem_util)
        promNodeMemDf = pd.DataFrame(promNodeMemDict)

        promNodeRbUtils = self.promApi.getReceiveBandwidthUtilizationByNode(self.duration)
        promNodeRbDict = map(lambda b: {"hostname": b["metric"]["node"], "receive_bytes": float(b["value"][1])}, promNodeRbUtils)
        # (id, hostname, receive_bytes)
        promNodeRbDf = pd.DataFrame(promNodeRbDict)

        promNodeTbUtils = self.promApi.getTransmitBandwidthUtilizationByNode(self.duration)
        promNodeTbDict = map(lambda b: {"hostname": b["metric"]["node"], "transmit_bytes": float(b["value"][1])}, promNodeTbUtils)
        # (id, hostname, transmit_bytes)
        promNodeTbDf = pd.DataFrame(promNodeTbDict)

        promNodeDf = pd.merge(promNodeCpuDf, promNodeMemDf, on=["hostname"], how="inner")
        promNodeDf = promNodeDf.merge(promNodeRbDf, on=["hostname"], how="inner")
        # (id, hostname, cpu_util, mem_util, receive_bytes, transmit_bytes)
        promNodeDf = promNodeDf.merge(promNodeTbDf, on=["hostname"], how="inner")

        # (id, hostname, ip, cpu_cap, mem_cap, cpu_util, mem_util, receive_bytes, transmit_bytes)
        nodeVertexDf = pd.merge(k8sNodeDf, promNodeDf, on=["hostname"], how='inner')
        # <-- [END] get node vertex information -->

        # <-- get node edge information -->
        def ipToHostname(ip): return nodeVertexDf[nodeVertexDf['ip'] == ip].iloc[0]['hostname']

        promReceiveBandwidths = self.promApi.getReceiveBandwidthBetweenNodes(self.duration)
        promRb = map(lambda b: {"src": ipToHostname(b["metric"]["source"]), "dst": b["metric"]
                                ["instance"], "receive_bytes": float(b["value"][1])}, promReceiveBandwidths)
        # (id, src, dst, receive_bytes)
        promRbDf = pd.DataFrame(promRb)

        promTransmitBandwidths = self.promApi.getTransmitBandwidthBetweenNodes(self.duration)
        promTb = map(lambda b: {"src": b["metric"]["instance"], "dst": ipToHostname(
            b["metric"]["destination"]), "transmit_bytes": float(b["value"][1])}, promTransmitBandwidths)
        # (id, src, dst, transmit_bytes)
        promTbDf = pd.DataFrame(promTb)

        promPingLatencies = self.promApi.getPingLatenciesBetweenNodes()
        promPl = map(lambda b: {"src": b["metric"]["instance"], "dst": ipToHostname(
            b["metric"]["destination"]), "latency_microseconds": float(b["value"][1])}, promPingLatencies)
        # (id, src, dst, latency_microseconds)
        promPlDf = pd.DataFrame(promPl)

        nodeEdgeDf = pd.merge(promRbDf, promTbDf, on=["src", "dst"], how="inner")
        # (id, src, dst, receive_bytes, transmit_bytes, latency_microseconds)
        nodeEdgeDf = nodeEdgeDf.merge(promPlDf, on=["src", "dst"], how="inner")
        # <-- [END] get node edge information -->

        return nodeVertexDf, nodeEdgeDf

    def getPods(self):
        """
        Return:
        - podVertexDf
            - type: pd.DataFrame
            - columns: (id, nodename, name, cpu_util, mem_util, receive_bytes, transmit_bytes)
        - podInNodeEdgeDf
            - type: pd.DataFrame
            - columns: (id, pode, node)
        """
        # <-- get pod vertex information -->
        k8sPods = self.k8sApi.getPods(self.namespace)

        k8sPodInfos = []
        for k8sPod in k8sPods:
            k8sPodInfo = defaultdict(int)
            if k8sPod.spec.node_name is None:
                continue # skip pod that is not scheduled
            k8sPodInfo["name"] = k8sPod.metadata.name
            k8sPodInfo["nodename"] = k8sPod.spec.node_name
            k8sPodInfos.append(k8sPodInfo)
        
        # (id, name, nodename)
        k8sPodDf = pd.DataFrame(k8sPodInfos)

        promPodCpuUtils = self.promApi.getCpuUtilizationByPod(self.namespace, self.duration)
        promPodCpuDict = map(lambda b: {"name": b["metric"]["pod"], "cpu_util": float(b["value"][1])}, promPodCpuUtils)
        # (id, name, cpu_util)
        promPodCpuDf = pd.DataFrame(promPodCpuDict)

        promPodMemUtils = self.promApi.getMemoryUtilizationByPod(self.namespace)
        promPodMemDict = map(lambda b: {"name": b["metric"]["pod"], "mem_util": float(b["value"][1])}, promPodMemUtils)
        # (id, name, mem_util)
        promPodMemDf = pd.DataFrame(promPodMemDict)

        promPodRbUtils = self.promApi.getReceiveBandwidthUtilizationByPod(self.namespace, self.duration)
        promPodRbDict = map(lambda b: {"name": b["metric"]["pod"], "receive_bytes": float(b["value"][1])}, promPodRbUtils)
        # (id, name, receive_bytes)
        promPodRbDf = pd.DataFrame(promPodRbDict)

        promPodTbUtils = self.promApi.getTransmitBandwidthUtilizationByPod(self.namespace, self.duration)
        promPodTbDict = map(lambda b: {"name": b["metric"]["pod"], "transmit_bytes": float(b["value"][1])}, promPodTbUtils)
        # (id, name, transmit_bytes)
        promPodTbDf = pd.DataFrame(promPodTbDict)

        promPodDf = pd.merge(promPodCpuDf, promPodMemDf, on=["name"], how="outer")
        promPodDf = promPodDf.merge(promPodRbDf, on=["name"], how="outer")
        # (id, name, cpu_util, mem_util, receive_bytes, transmit_bytes)
        promPodDf = promPodDf.merge(promPodTbDf, on=["name"], how="outer")
        # (id, nodename, name, cpu_util, mem_util, receive_bytes, transmit_bytes)
        podVertexDf = pd.merge(k8sPodDf, promPodDf, on=["name"], how="inner")
        # <-- [END] get pod vertex information -->

        # <-- get pod-in-node edge information -->
        podInNodeEdge = map(lambda pod: {"node": pod["nodename"],"pod": pod["name"]}, k8sPodInfos)
        podInNodeEdgeDf = pd.DataFrame(podInNodeEdge)
        # <-- [END] get pod-in-node edge information -->

        return podVertexDf, podInNodeEdgeDf

    def getServices(self, podVertexDf):
        """Return:
        - serviceVertexDf
            - type: pd.DataFrame
            - columns: (id, name, pods, cpu_util, mem_util, receive_bytes, transmit_bytes)
        - serviceEdgeDf
            - type: pd.DataFrame
            - columns: (id, src, dst, receive_bytes, transmit_bytes, latency_microseconds)
        - serviceSelectPodEdgeDf
            - type: pd.DataFrame
            - columns: (id, service, pod)
        """
        # <-- get service vertex information -->
        k8sServices = self.k8sApi.getServices(self.namespace)
        k8sPods = self.k8sApi.getPods(self.namespace)

        k8sServiceInfos = []
        for k8sService in k8sServices:
            k8sServiceInfo = defaultdict(list)
            k8sServiceInfo["name"] = k8sService.metadata.name
            # get pod selected by service
            selector = k8sService.spec.selector
            if selector is not None:
                # check whether dict1 is subset of dict2.
                def isSubset(dict1, dict2): return all(item in dict2.items() for item in dict1.items())
                selectedPods = list(filter(lambda pod: isSubset(selector, pod.metadata.labels), k8sPods))
                for pod in selectedPods:
                    if pod.spec.node_name is None:
                        continue # skip pod that is not scheduled
                    k8sServiceInfo["pods"].append(pod.metadata.name)
            k8sServiceInfos.append(k8sServiceInfo)
        
        # (name, nodename, cpu_util, mem_util, receive_bytes, transmit_bytes)
        indexer = podVertexDf.set_index("name")
        for k8sServiceInfo in k8sServiceInfos:
            if len(k8sServiceInfo["pods"]) == 0:
                k8sServiceInfo["cpu_util"] = 0
                k8sServiceInfo["mem_util"] = 0
                k8sServiceInfo["receive_bytes"] = 0
                k8sServiceInfo["transmit_bytes"] = 0
            else:
                valid_names = indexer.index.intersection(k8sServiceInfo["pods"])
                k8sServiceInfo["cpu_util"] = indexer.loc[valid_names].fillna(0)["cpu_util"].sum()
                k8sServiceInfo["mem_util"] = indexer.loc[valid_names].fillna(0)["mem_util"].sum()
                k8sServiceInfo["receive_bytes"] = indexer.loc[valid_names].fillna(0)["receive_bytes"].sum()
                k8sServiceInfo["transmit_bytes"] = indexer.loc[valid_names].fillna(0)["transmit_bytes"].sum()

        # (id, name, pods, cpu_util, mem_util, receive_bytes, transmit_bytes)
        serviceVertexDf = pd.DataFrame(k8sServiceInfos)
        # <-- [END] get service vertex information -->

        # <-- get service edge information -->
        promServiceRb = self.promApi.getReceiveBandwidthBetweenServices(self.namespace, self.duration)
        promServiceRbDict = map(lambda b: {"src": b["metric"]["source_workload"], "dst": b["metric"]
                                ["destination_workload"], "receive_bytes": float(b["value"][1])}, promServiceRb)
        promServiceRbDf = pd.DataFrame(promServiceRbDict)

        promServiceTb = self.promApi.getTransmitBandwidthBetweenServices(self.namespace, self.duration)
        promServiceTbDict = map(lambda b: {"src": b["metric"]["source_workload"], "dst": b["metric"]
                                ["destination_workload"], "transmit_bytes": float(b["value"][1])}, promServiceTb)
        promServiceTbDf = pd.DataFrame(promServiceTbDict)

        promServiceDur = self.promApi.getDurationBetweenServices(self.namespace, self.duration)
        promServiceDurDict = map(lambda b: {"src": b["metric"]["source_workload"], "dst": b["metric"]
                               ["destination_workload"], "duration": float(b["value"][1])}, promServiceDur)
        promServiceDurDf = pd.DataFrame(promServiceDurDict)

        serviceEdgeDf = pd.merge(promServiceRbDf, promServiceTbDf, on=["src", "dst"], how="inner")
        # (id, src, dst, receive_bytes, transmit_bytes)
        serviceEdgeDf = serviceEdgeDf.merge(promServiceDurDf, on=["src", "dst"], how="inner")
        # <-- [END] get service edge information -->

        # <-- get service-select-pod edge information -->
        serviceSelectPodEdge = []
        for serviceName, podNames in zip(serviceVertexDf["name"], serviceVertexDf["pods"]):
            for podName in podNames:
                serviceSelectPodEdge.append({"service": serviceName, "pod": podName})
        # (id, service, pod)
        serviceSelectPodEdgeDf = pd.DataFrame(serviceSelectPodEdge)

        return serviceVertexDf, serviceEdgeDf, serviceSelectPodEdgeDf

    def getGraph(self):
        nodeVertexDf, nodeEdgeDf = self.getNodes()
        podVertexDf, podInNodeEdgeDf = self.getPods()
        serviceVertexDf, serviceEdgeDf, serviceSelectPodEdgeDf = self.getServices(podVertexDf)
        
        return {
            "nodeVertexDf": nodeVertexDf,
            "nodeEdgeDf": nodeEdgeDf,
            "serviceVertexDf": serviceVertexDf,
            "serviceEdgeDf": serviceEdgeDf,
            "podVertexDf": podVertexDf,
            "podInNodeEdgeDf": podInNodeEdgeDf,
            "serviceSelectPodEdgeDf": serviceSelectPodEdgeDf,
        }
        
    def getPowerConsumptions(self):
        powerConsumptionJule = self.promApi.getPowerConsumptionByNode(self.duration)
        powerConsumptions = []
        for data in powerConsumptionJule:
            powerConsumptions.append({"nodename": data["metric"]["instance"], "value": float(data["value"][1])})
        return powerConsumptions

    def getServicenameByPodlabels(self, podlabels: List[str]):
        podlabel_dict = {podlabel.split("=")[0]: podlabel.split("=")[1] for podlabel in podlabels}
        k8sServices = self.k8sApi.getServices()
        for k8sService in k8sServices:
            # get pod selected by service
            selector = k8sService.spec.selector
            if selector is not None:
                # check whether dict1 is subset of dict2.
                def isSubset(dict1, dict2): return all(item in dict2.items() for item in dict1.items())
                if isSubset(selector, podlabel_dict):
                    return k8sService.metadata.name


if __name__ == "__main__":
    k8sApi = initK8sApi(
        config_file='/home/dpnm/projects/rl-server/kube-config.yaml')
    promApi = initPromApi(
        url='http://sfc-testbed.duckdns.org:31237/prometheus', disable_ssl=True)
    collector = Collector(k8sApi, promApi)
    graph = collector.getGraph()
    print(graph)