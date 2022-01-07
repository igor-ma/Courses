import requests

json_ts = {"size": 120, "year": 2001, "garage": 2}
url = 'http://127.0.0.1:5000/house/pricing/'
auth = requests.auth.HTTPBasicAuth('admin', '123')
r = requests.post(url, json=json_ts, auth=auth)
print(r.json()) #{'price': 192448.63169820444}