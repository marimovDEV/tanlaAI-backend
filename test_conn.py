import socket
import sys

def test_dns():
    host = 'api.telegram.org'
    print(f"Testing DNS resolution for {host}...")
    try:
        addr = socket.gethostbyname(host)
        print(f"Successfully resolved {host} to {addr}")
    except Exception as e:
        print(f"DNS Resolution failed: {e}")

def test_connect():
    host = '149.154.166.110' # api.telegram.org IP
    port = 443
    print(f"Testing TCP connection to {host}:{port}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect((host, port))
        print(f"Successfully connected to {host}")
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        s.close()

if __name__ == "__main__":
    test_dns()
    print("-" * 20)
    test_connect()
