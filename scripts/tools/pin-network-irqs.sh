#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: sudo $0 <pin|restore> <interface> [reserved-cpu-list]" >&2
    echo "example: sudo $0 pin ens66 0-7" >&2
    exit 2
}

[[ $# -ge 2 ]] || usage
action=$1
interface=$2
cpu_list=${3:-}
state_dir=${STATE_DIR:-/var/lib/iris-mpc}
state_file="${state_dir}/irq-affinity-${interface}.state"

[[ ${EUID} -eq 0 ]] || {
    echo "this script must run as root" >&2
    exit 1
}
[[ -d "/sys/class/net/${interface}" ]] || {
    echo "network interface ${interface} does not exist" >&2
    exit 1
}

expand_cpu_list() {
    local part first last cpu
    local -a expanded=()
    IFS=',' read -ra parts <<<"$1"
    for part in "${parts[@]}"; do
        if [[ ${part} == *-* ]]; then
            first=${part%-*}
            last=${part#*-}
            for ((cpu = first; cpu <= last; cpu++)); do
                expanded+=("${cpu}")
            done
        else
            expanded+=("${part}")
        fi
    done
    printf '%s\n' "${expanded[@]}"
}

case ${action} in
pin)
    [[ -n ${cpu_list} ]] || usage
    [[ ! -e ${state_file} ]] || {
        echo "${state_file} already exists; restore before pinning again" >&2
        exit 1
    }
    mapfile -t cpus < <(expand_cpu_list "${cpu_list}")
    ((${#cpus[@]} > 0)) || {
        echo "reserved CPU list is empty" >&2
        exit 1
    }
    for cpu in "${cpus[@]}"; do
        [[ -d "/sys/devices/system/cpu/cpu${cpu}" ]] || {
            echo "CPU ${cpu} does not exist" >&2
            exit 1
        }
    done
    mapfile -t irqs < <(
        awk -v queue_prefix="${interface}-Tx-Rx-" \
            'index($0, queue_prefix) { gsub(":", "", $1); print $1 }' \
            /proc/interrupts
    )
    ((${#irqs[@]} > 0)) || {
        echo "no ${interface} Tx/Rx queue IRQs found" >&2
        exit 1
    }

    install -d "${state_dir}"
    irqbalance_active=0
    if systemctl is-active --quiet irqbalance; then
        irqbalance_active=1
        systemctl stop irqbalance
    fi
    printf 'irqbalance_active %s\n' "${irqbalance_active}" >"${state_file}"
    for index in "${!irqs[@]}"; do
        irq=${irqs[index]}
        cpu_index=$((index % ${#cpus[@]}))
        affinity=$(<"/proc/irq/${irq}/smp_affinity_list")
        printf 'irq %s %s\n' "${irq}" "${affinity}" >>"${state_file}"
        printf '%s\n' "${cpus[cpu_index]}" >"/proc/irq/${irq}/smp_affinity_list"
    done
    echo "pinned ${#irqs[@]} ${interface} queue IRQs across CPUs ${cpu_list}"
    ;;
restore)
    [[ -f ${state_file} ]] || {
        echo "missing saved state ${state_file}" >&2
        exit 1
    }
    irqbalance_active=0
    while read -r kind first second; do
        case ${kind} in
        irqbalance_active) irqbalance_active=${first} ;;
        irq) printf '%s\n' "${second}" >"/proc/irq/${first}/smp_affinity_list" ;;
        esac
    done <"${state_file}"
    if [[ ${irqbalance_active} == 1 ]]; then
        systemctl start irqbalance
    fi
    rm "${state_file}"
    echo "restored ${interface} IRQ affinities"
    ;;
*) usage ;;
esac
