from src.env.vnfs.const import env_types, vnf_load
from src.env.vnfs.vnf import VNF
from src.apis.k8s import K8sApi

vnf_name_types = [
    {"name": "vnf-account-0", "type": "vnf-account"},
    {"name": "vnf-account-1", "type": "vnf-account"},
    {"name": "vnf-firewall-0", "type": "vnf-firewall"},
    {"name": "vnf-host-id-injection-0", "type": "vnf-host-id-injection"},
    {"name": "vnf-ids-0", "type": "vnf-ids"},
    {"name": "vnf-nat-0", "type": "vnf-nat"},
    {"name": "vnf-observer-0", "type": "vnf-observer"},
    {"name": "vnf-registry-0", "type": "vnf-registry"},
    {"name": "vnf-session-0", "type": "vnf-session"},
    {"name": "vnf-tcp-optimizer-0", "type": "vnf-tcp-optimizer"},
]


def create_vnfs(k8sApi: K8sApi, namespace: str):
    image = "euidong/vnf-scc-sfc:prod-0.0.2"
    return [
        VNF(
            k8sApi=k8sApi,
            namespace=namespace,
            image=image,
            name=nt["name"],
            envs=dict(
                **env_types["cpu"][vnf_load[nt["type"]]["cpu"]],
                **env_types["disk"][vnf_load[nt["type"]]["disk"]],
                **env_types["memory"][vnf_load[nt["type"]]["memory"]],
            )
        )
        for nt in vnf_name_types
    ]


# {
#                 "CPU_OPS": "250",
#                 "CPU_WORKER": "1",
#                 "CPU_LIMIT": "30",
#                 "MEM_OPS": "250",
#                 "MEM_WORKER": "1",
#                 "MEM_BYTES": "50000",
#                 "DIO_OPS": "250",
#                 "DIO_WORKER": "1",
#                 "DIO_BYTES": "5000000",
#             }

