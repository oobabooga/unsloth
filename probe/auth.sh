# sourced: sets TOKEN for a studio backend on $PORT
studio_token () {
  local port="$1"
  local boot
  boot="$(cat "$UNSLOTH_STUDIO_HOME/auth/.bootstrap_password" 2>/dev/null)"
  if [ -z "$boot" ]; then echo "NO BOOTSTRAP PASSWORD" >&2; return 1; fi
  local resp
  resp="$(curl -s "http://127.0.0.1:$port/api/auth/login" -H 'Content-Type: application/json' \
    -d "{\"username\": \"unsloth\", \"password\": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$boot")}")"
  TOKEN="$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("access_token",""))')"
  local must
  must="$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("must_change_password",False))')"
  if [ "$must" = "True" ]; then
    resp="$(curl -s -X POST "http://127.0.0.1:$port/api/auth/change-password" \
      -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
      -d "{\"current_password\": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$boot"), \"new_password\": \"UnslothProbe12345\"}")"
    TOKEN="$(printf '%s' "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("access_token",""))')"
  fi
  [ -n "$TOKEN" ] || { echo "NO TOKEN: $resp" >&2; return 1; }
  export TOKEN
}
