import requests

NIST_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"

params = {
    "spectra": "Li I",
    "limits_type": "0",
    "low_w": "200",
    "upp_w": "900",
    "unit": "1",
    "format": "3",  # Tab-delimited
    "line_out": "0",
    "remove_j": "on",
    "page_size": "0",
    "show_obs_wl": "1",
    "show_calc_wl": "1",
    "intens_out": "on"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

resp = requests.get(NIST_URL, params=params, headers=headers, timeout=15)
print(f"Status Code: {resp.status_code}")
print("--- FIRST 500 CHARACTERS OF RESPONSE ---")
print(resp.text[:500])