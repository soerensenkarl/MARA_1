import os
import subprocess
import time

HOST = "192.168.80.3"      # your Spot IP
USER = "user"             # your username
PASS = "password"     # your password

def run_module(modname):
    print(f"\n=== Running {modname} ===")
    env = os.environ.copy()
    env["BOSDYN_CLIENT_USERNAME"] = USER
    env["BOSDYN_CLIENT_PASSWORD"] = PASS
    # If you also use certs/app tokens, set env here too.
    subprocess.run(["py", "-m", modname, HOST], env=env, check=True)

if __name__ == "__main__":
    run_module("arm_simple")
    time.sleep(2)
    run_module("arm_simple2")
    print("\nSequence complete.")
