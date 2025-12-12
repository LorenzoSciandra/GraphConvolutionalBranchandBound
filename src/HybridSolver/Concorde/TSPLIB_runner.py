import subprocess
import re
from pathlib import Path

# === Configuration ===
TIME_LIMIT = 428996  # seconds (10 minutes)
root_dir = Path(__file__).parent.resolve()
build_dir = root_dir / "build"
tsplib_dir = root_dir / "examples" / "TSPLIB"
output_file = root_dir / "results.txt"

# === Step 1: Run compile.sh ===
print("Running compile.sh...")
subprocess.run(["sh", "compile.sh"], cwd=root_dir, check=True)

# === Step 2: Prepare output file ===
with open(output_file, "w") as f:
    f.write("instance,num_bbnodes,total_time\n")

# === Step 3: Run Concorde with timeout ===
print("Running Concorde on all .tsp files (time limit: ", round(TIME_LIMIT/60, 2), "min each)...")

for tsp_file in sorted(tsplib_dir.glob("*.tsp")):
    print(f"Solving {tsp_file.name}...")

    try:
        result = subprocess.run(
            ["./concorde-bin", f"../examples/TSPLIB/{tsp_file.name}"],
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIME_LIMIT,
            check=False  # allow nonzero exit codes
        )
        output = result.stdout

        # Extract Number of bbnodes and Total Running Time
        nodes_match = re.search(r"Number of bbnodes:\s*(\d+)", output)
        time_match = re.search(r"Total Running Time:\s*([\d.]+)", output)

        if nodes_match and time_match:
            num_bbnodes = nodes_match.group(1)
            total_time = time_match.group(1)
            print(f"bbnodes: {num_bbnodes}, time: {total_time}s")

            with open(output_file, "a") as f:
                f.write(f"{tsp_file.stem},{num_bbnodes},{total_time}\n")
        else:
            print(f"Could not parse output for {tsp_file.name}")

    except subprocess.TimeoutExpired:
        print(f"Timeout after {TIME_LIMIT/60:.1f} min — skipping {tsp_file.name}")
        with open(output_file, "a") as f:
            f.write(f"{tsp_file.stem},TIMEOUT,>\n")

print(f"\nAll instances processed. Results saved to '{output_file}'.")
