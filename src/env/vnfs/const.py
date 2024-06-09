# ### CPU

# |        | CPU_OPERATION_NUM | CPU_LIMIT(%) |
# |--------|-------------------|--------------|
# | High   | 1000              | 10           |
# | Middle | 250               | 10           |
# | Low    | 100               | 10           |


# ### Memory

# |        | MEM_OPERATION_NUM | MEM_BYTES(b) |
# |--------|-------------------|--------------|
# | High   | 1000              | 100000       |
# | Middle | 250               | 50000        |
# | Low    | 100               | 10000        |

# ### Disk

# |        | DISK_OPERATION_NUM | DISK_BYTES(b) |
# |--------|--------------------|---------------|
# | High   | 1000               | 10000000      |
# | Middle | 250                | 5000000       |
# | Low    | 100                | 2000000       |

env_types = {
    "cpu": {
        "high": {
            "CPU_OPS": "1000",
            "CPU_WORKER": "1",
            "CPU_LIMIT": "10",
        },
        "middle": {
            "CPU_OPS": "250",
            "CPU_WORKER": "1",
            "CPU_LIMIT": "10",
        },
        "low": {
            "CPU_OPS": "100",
            "CPU_WORKER": "1",
            "CPU_LIMIT": "10",
        },
    },
    "memory": {
        "high": {
            "MEM_OPS": "1000",
            "MEM_WORKER": "1",
            "MEM_BYTES": "100000",
        },
        "middle": {
            "MEM_OPS": "250",
            "MEM_WORKER": "1",
            "MEM_BYTES": "50000",
        },
        "low": {
            "MEM_OPS": "100",
            "MEM_WORKER": "1",
            "MEM_BYTES": "10000",
        },
    },
    "disk": {
        "high": {
            "DIO_OPS": "1000",
            "DIO_WORKER": "1",
            "DIO_BYTES": "10000000",
        },
        "middle": {
            "DIO_OPS": "250",
            "DIO_WORKER": "1",
            "DIO_BYTES": "5000000",
        },
        "low": {
            "DIO_OPS": "100",
            "DIO_WORKER": "1",
            "DIO_BYTES": "2000000",
        },
    },
}

## VNF's load

# |                   | CPU  | Disk | Memory |
# |-------------------|------|------|--------|
# | Account           | mid  | mid  | mid    |
# | Firewall          | high | low  | high   |
# | Host ID injection | high | high | low    |
# | IDS               | high | high | high   |
# | NAT               | high | low  | low    |
# | Observer          | low  | low  | low    |
# | Registry          | low  | high | high   |
# | Session           | low  | high | low    |
# | TCP optimizer     | low  | low  | high   |

vnf_load = {
    "vnf-account": {
        "cpu": "middle",
        "disk": "middle",
        "memory": "middle",
    },
    "vnf-firewall": {
        "cpu": "high",
        "disk": "low",
        "memory": "high",
    },
    "vnf-host-id-injection": {
        "cpu": "high",
        "disk": "high",
        "memory": "low",
    },
    "vnf-ids": {
        "cpu": "high",
        "disk": "high",
        "memory": "high",
    },
    "vnf-nat": {
        "cpu": "high",
        "disk": "low",
        "memory": "low",
    },
    "vnf-observer": {
        "cpu": "low",
        "disk": "low",
        "memory": "low",
    },
    "vnf-registry": {
        "cpu": "low",
        "disk": "high",
        "memory": "high",
    },
    "vnf-session": {
        "cpu": "low",
        "disk": "high",
        "memory": "low",
    },
    "vnf-tcp-optimizer": {
        "cpu": "low",
        "disk": "low",
        "memory": "high",
    },
}