import requests

from its_common import get_api_key

url = "https://openapi.its.go.kr:9443/cctvInfo"

params = {
    "apiKey": get_api_key(),
    "type": "all",
    "cctvType": "4",
    "minX": "127.00",
    "maxX": "127.30",
    "minY": "37.30",
    "maxY": "37.60",
    "getType": "json",
}

res = requests.get(url, params=params, timeout=30)

print(res.status_code)
print(res.text[:3000])
