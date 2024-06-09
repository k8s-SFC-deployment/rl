from prometheus_api_client import PrometheusConnect


def initApi(url, disable_ssl):
    return PromApi(url, disable_ssl)


class PromApi:
    api: PrometheusConnect

    def __init__(self, url, disable_ssl):
        self.api = PrometheusConnect(url=url, disable_ssl=disable_ssl)

    def getCpuUtilizationByNode(self, time_range='1m'):
        query = 'sum(rate(node_cpu_seconds_total{mode="idle"}[%s])) by (instance, node)' % time_range
        result = self.api.custom_query(query=query)
        return result

    def getMemoryUtilizationByNode(self):
        query = 'sum(node_memory_MemTotal_bytes - (node_memory_MemFree_bytes + node_memory_Buffers_bytes + node_memory_Cached_bytes)) by (instance, node) / sum(node_memory_MemTotal_bytes) by (instance, node) * 100'
        result = self.api.custom_query(query=query)
        return result

    def getReceiveBandwidthUtilizationByNode(self, time_range='1m'):
        query = 'sum(rate(node_network_receive_bytes_total[%s])) by (instance, node)' % time_range
        result = self.api.custom_query(query=query)
        return result

    def getTransmitBandwidthUtilizationByNode(self, time_range='1m'):
        query = 'sum(rate(node_network_transmit_bytes_total[%s])) by (instance, node)' % time_range
        result = self.api.custom_query(query=query)
        return result

    def getPingLatenciesBetweenNodes(self):
        query = 'sum (nmbn_current_latency_microseconds) by (instance, destination)'
        result = self.api.custom_query(query=query)
        return result

    def getReceiveBandwidthBetweenNodes(self, time_range='1m'):
        query = 'sum(rate(nmbn_bandwidth_receive_bytes[%s])) by (instance, source)' % time_range
        result = self.api.custom_query(query=query)
        return result

    def getTransmitBandwidthBetweenNodes(self, time_range='1m'):
        query = 'sum(rate(nmbn_bandwidth_transmit_bytes[%s])) by (instance, destination)' % time_range
        result = self.api.custom_query(query=query)
        return result

    def getReceiveBandwidthBetweenServices(self, namespace, time_range='1m'):
        query = 'avg (rate(istio_response_bytes_sum{reporter="destination", source_workload_namespace="%s",destination_workload_namespace="%s"}[%s])) by (source_workload, destination_workload)' % (namespace, namespace, time_range)
        result = self.api.custom_query(query=query)
        return result

    def getTransmitBandwidthBetweenServices(self, namespace, time_range='1m'):
        query = 'sum(rate(istio_request_bytes_sum{reporter="destination", source_workload_namespace="%s",destination_workload_namespace="%s"}[%s])) by (source_workload, destination_workload)' % (namespace, namespace, time_range) 
        result = self.api.custom_query(query=query)
        return result

    def getDurationBetweenServices(self, namespace, time_range='1m'):
        query = 'sum(rate(istio_request_duration_milliseconds_bucket{reporter="destination", source_workload_namespace="%s",destination_workload_namespace="%s"}[%s])) by (source_workload, destination_workload)' % (namespace, namespace, time_range)
        result = self.api.custom_query(query=query)
        return result

    def getCpuUtilizationByPod(self, namespace, time_range='1m'):
        query = 'sum(rate(container_cpu_usage_seconds_total{container!="POD",container!="", namespace="%s"}[%s])) by (pod)' % (namespace, time_range)
        result = self.api.custom_query(query=query)
        return result

    def getMemoryUtilizationByPod(self, namespace):
        # base is node_memory
        query = 'sum(container_memory_usage_bytes{container!="POD",container!="", namespace="%s"}) by (pod, instance)' % namespace
        pod_memory = self.api.custom_query(query=query)

        query = 'sum(node_memory_MemTotal_bytes) by (node)'
        node_tot_memory = self.api.custom_query(query=query)

        result = []
        for pod in pod_memory:
            for node in node_tot_memory:
                if pod['metric']['instance'] == node['metric']['node']:
                    result.append({
                        'metric': {
                            'pod': pod['metric']['pod'],
                            'instance': node['metric']['node']
                        },
                        'value': [
                            pod['value'][0],
                            float(pod['value'][1]) / float(node['value'][1]) * 100
                        ]
                    })
        return result

    def getReceiveBandwidthUtilizationByPod(self, namespace, time_range='1m'):
        query = 'sum(rate(container_network_receive_bytes_total{pod!="", namespace="%s"}[%s])) by (pod)' % (namespace, time_range)
        result = self.api.custom_query(query=query)
        return result
    
    def getTransmitBandwidthUtilizationByPod(self, namespace, time_range='1m'):
        query = 'sum(rate(container_network_transmit_bytes_total{pod!="", namespace="%s"}[%s])) by (pod)' % (namespace, time_range)
        result = self.api.custom_query(query=query)
        return result

    def getPowerConsumptionByNode(self, time_range='1m', scrape_interval=10):
        query = 'sum_over_time(powertop_baseline_power_count[%s]) * %d' % (time_range, scrape_interval)
        result = self.api.custom_query(query=query)
        return result

if __name__ == '__main__':
    promApi = initApi(
        url='http://sfc-testbed.duckdns.org:31237/prometheus', disable_ssl=True)
    print(promApi.getCPUUtilizationByPods())
