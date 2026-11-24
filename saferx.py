import httpx

resp = httpx.post(
    "https://saferx.online/api/developers/keys/free",
    json={"email": "phooiechennie@gmail.com"},
)
print(resp.json()["message"])  # "Verification code sent to your email"
