#!/usr/bin/env bash

# Shared network selection and validation for Go2 DDS tools.

GO2_ETH_IF_DEFAULT="${GO2_ETH_IF:-eth0}"
GO2_WIFI_IF_DEFAULT="${GO2_WIFI_IF:-}"

go2_is_wireless_interface() {
  local interface="$1"
  [[ -d "/sys/class/net/${interface}/wireless" ]] && return 0
  [[ -L "/sys/class/net/${interface}/phy80211" ]] && return 0
  return 1
}

go2_first_wireless_interface() {
  local path
  for path in /sys/class/net/*; do
    [[ -e "${path}" ]] || continue
    if go2_is_wireless_interface "$(basename "${path}")"; then
      basename "${path}"
      return 0
    fi
  done
  return 1
}

go2_interface_ipv4() {
  local interface="$1"
  ip -4 -o addr show dev "${interface}" scope global 2>/dev/null | awk '{print $4}' | paste -sd, -
}

go2_interface_operstate() {
  local interface="$1"
  cat "/sys/class/net/${interface}/operstate" 2>/dev/null || printf 'unknown\n'
}

go2_interface_carrier() {
  local interface="$1"
  cat "/sys/class/net/${interface}/carrier" 2>/dev/null || printf 'unknown\n'
}

go2_interface_has_multicast() {
  local interface="$1"
  local flags
  flags="$(cat "/sys/class/net/${interface}/flags" 2>/dev/null || true)"
  [[ -n "${flags}" ]] || return 1
  (( (flags & 0x1000) != 0 ))
}

go2_interface_ready() {
  local interface="$1"
  [[ -d "/sys/class/net/${interface}" ]] || return 1
  [[ "$(go2_interface_carrier "${interface}")" != "0" ]] || return 1
  [[ -n "$(go2_interface_ipv4 "${interface}")" ]] || return 1
  go2_interface_has_multicast "${interface}"
}

go2_resolve_network_interface() {
  local selector="${1:-ethernet}"
  local explicit_interface="${2:-}"
  local interface=""

  if [[ -n "${explicit_interface}" ]]; then
    printf '%s\n' "${explicit_interface}"
    return 0
  fi

  case "${selector}" in
    ethernet)
      interface="${GO2_ETH_IF_DEFAULT}"
      ;;
    wifi|wireless)
      interface="${GO2_WIFI_IF_DEFAULT}"
      if [[ -z "${interface}" ]]; then
        interface="$(go2_first_wireless_interface || true)"
      fi
      ;;
    auto)
      if [[ -n "${GO2_NET_IF:-}" ]] && go2_interface_ready "${GO2_NET_IF}"; then
        interface="${GO2_NET_IF}"
      elif go2_interface_ready "${GO2_ETH_IF_DEFAULT}"; then
        interface="${GO2_ETH_IF_DEFAULT}"
      else
        local wifi_candidate="${GO2_WIFI_IF_DEFAULT}"
        if [[ -z "${wifi_candidate}" ]]; then
          wifi_candidate="$(go2_first_wireless_interface || true)"
        fi
        if [[ -n "${wifi_candidate}" ]] && go2_interface_ready "${wifi_candidate}"; then
          interface="${wifi_candidate}"
        else
          echo "No ready Go2 transport found." >&2
          return 2
        fi
      fi
      ;;
    *)
      interface="${selector}"
      ;;
  esac

  if [[ -z "${interface}" ]]; then
    echo "No interface found for transport '${selector}'." >&2
    return 2
  fi
  printf '%s\n' "${interface}"
}

go2_print_network_status() {
  local interface="$1"
  if [[ ! -d "/sys/class/net/${interface}" ]]; then
    printf 'interface=%s exists=no\n' "${interface}"
    return 1
  fi
  local transport="ethernet"
  if go2_is_wireless_interface "${interface}"; then
    transport="wifi"
  fi
  printf 'interface=%s\n' "${interface}"
  printf 'transport=%s\n' "${transport}"
  printf 'operstate=%s\n' "$(go2_interface_operstate "${interface}")"
  printf 'carrier=%s\n' "$(go2_interface_carrier "${interface}")"
  printf 'ipv4=%s\n' "$(go2_interface_ipv4 "${interface}")"
  if go2_interface_has_multicast "${interface}"; then
    printf 'multicast=yes\n'
  else
    printf 'multicast=no\n'
  fi
}

go2_validate_network_interface() {
  local interface="$1"
  if [[ ! -d "/sys/class/net/${interface}" ]]; then
    echo "Network interface '${interface}' does not exist." >&2
    return 2
  fi
  if [[ "$(go2_interface_carrier "${interface}")" == "0" ]]; then
    echo "Network interface '${interface}' has no link." >&2
    return 2
  fi
  if [[ -z "$(go2_interface_ipv4 "${interface}")" ]]; then
    echo "Network interface '${interface}' has no global IPv4 address." >&2
    return 2
  fi
  if ! go2_interface_has_multicast "${interface}"; then
    echo "Network interface '${interface}' is not multicast-capable." >&2
    return 2
  fi
}
