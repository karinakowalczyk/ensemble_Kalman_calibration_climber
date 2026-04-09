#!/usr/bin/env julia
"""
CLIMBER-X Job Submission Diagnostic Script

This script helps identify issues with job submission by testing
different command construction methods and comparing them to the
working bash script approach.

Run this interactively to diagnose submission problems:
    julia -i diagnose_submission.jl
"""

using Dates

# ============================================
# CONFIGURATION
# ============================================

const CLIMBER_X_DIR = "/home/karinako/climber-x"
const RUNME_SCRIPT = joinpath(CLIMBER_X_DIR, "runme")

# Test parameters (middle of prior ranges)
const TEST_PARAMS = Dict{String, Any}(
    # Ocean parameters
    "ocn.diff_dia_min" => 1.0e-5,
    "ocn.drag_topo_fac" => 3.0,
    "ocn.slope_max" => 1.0e-3,
    "ocn.diff_iso" => 1500.0,
    "ocn.diff_gm" => 1500.0,
    "ocn.diff_dia_max" => 1.5e-4,
    # Control parameters
    "ctl.nyears" => 10000,  # 10000 year run
    "ctl.co2_const" => 190,
    "ctl.fake_geo_const_file" => "input/geo_ice_tarasov_12ka.nc",
    "ctl.fake_ice_const_file" => "input/geo_ice_tarasov_12ka.nc",
    "ctl.restart_in_dir" => "/home/karinako/climber-x/output/DO/spinup_ensemble/CO2_190/restart_out/year_3000",
    "ocn.l_noise_fw" => "T",
    "ocn.noise_amp_fw" => 0.4
)

# ============================================
# DIAGNOSTIC FUNCTIONS
# ============================================

"""
Check basic prerequisites
"""
function check_prerequisites()
    println("="^60)
    println("CHECKING PREREQUISITES")
    println("="^60)
    
    # Check runme exists
    if isfile(RUNME_SCRIPT)
        println("✓ runme script exists: $RUNME_SCRIPT")
    else
        println("✗ runme script NOT FOUND: $RUNME_SCRIPT")
        return false
    end
    
    # Check it's executable
    try
        run(`test -x $RUNME_SCRIPT`)
        println("✓ runme is executable")
    catch
        println("✗ runme is NOT executable")
        return false
    end
    
    # Check Python
    try
        run(`python3 --version`)
        println("✓ Python3 available")
    catch
        println("✗ Python3 not found")
        return false
    end
    
    # Check runner module
    try
        run(`python3 -c "import runner"`)
        println("✓ runner module available")
    catch
        println("✗ runner module not found")
        println("  Install with: pip install --user git+https://github.com/alex-robinson/runner.git")
        return false
    end
    
    # Check SLURM
    try
        run(`which sbatch`)
        println("✓ sbatch available")
    catch
        println("✗ sbatch not found - not on cluster?")
        return false
    end
    
    println("\nAll prerequisites OK!")
    return true
end

"""
Generate bash command string (what the working bash script does)
"""
function generate_bash_style_command(output_dir; qos="standby", walltime="01:00:00", omp=32)
    param_str = join(["$(k)=$(v)" for (k, v) in TEST_PARAMS], " \\\n           ")
    
    cmd = """./runme -s -q $(qos) -w $(walltime) --omp $(omp) \\
        -o "$(output_dir)" \\
        -p $(param_str)"""
    
    return cmd
end

"""
Method 1: Use bash -c with string command (most like bash script)
"""
function test_method_bash_c(output_dir; dry_run=true, qos="standby", walltime="01:00:00", omp=32)
    println("\n" * "-"^60)
    println("METHOD 1: bash -c with string command")
    println("-"^60)
    
    # Build parameter string
    param_str = ""
    for (key, val) in TEST_PARAMS
        param_str *= " $(key)=$(val)"
    end
    
    cmd_str = """./runme -s -q $(qos) -w $(walltime) --omp $(omp) -o "$(output_dir)" -p$(param_str)"""
    
    println("\nCommand string:")
    println(cmd_str)
    
    if !dry_run
        original_dir = pwd()
        cd(CLIMBER_X_DIR)
        try
            println("\nExecuting...")
            output = read(`bash -c $cmd_str`, String)
            println("Output: $output")
            return output
        catch e
            println("Error: $e")
            return nothing
        finally
            cd(original_dir)
        end
    end
    
    return cmd_str
end

"""
Method 2: Use Cmd array with splatted parameters
"""
function test_method_cmd_array(output_dir; dry_run=true, qos="standby", walltime="01:00:00", omp=32)
    println("\n" * "-"^60)
    println("METHOD 2: Cmd array with parameters")
    println("-"^60)
    
    # Build as array
    cmd_args = String["./runme", "-s", "-q", qos, "-w", walltime, "--omp", string(omp), "-o", output_dir, "-p"]
    
    for (key, val) in TEST_PARAMS
        push!(cmd_args, "$(key)=$(val)")
    end
    
    println("\nCommand array:")
    for (i, arg) in enumerate(cmd_args)
        println("  [$i] $arg")
    end
    
    if !dry_run
        original_dir = pwd()
        cd(CLIMBER_X_DIR)
        try
            cmd = Cmd(cmd_args)
            println("\nExecuting: $cmd")
            output = read(cmd, String)
            println("Output: $output")
            return output
        catch e
            println("Error: $e")
            return nothing
        finally
            cd(original_dir)
        end
    end
    
    return join(cmd_args, " ")
end

"""
Method 3: Use backtick interpolation
"""
function test_method_backtick(output_dir; dry_run=true, qos="standby", walltime="01:00:00", omp=32)
    println("\n" * "-"^60)
    println("METHOD 3: Backtick interpolation")
    println("-"^60)
    
    # Build parameter list
    param_args = ["$(k)=$(v)" for (k, v) in TEST_PARAMS]
    
    println("\nParameter args: $param_args")
    println("\nThis will expand to:")
    println("  runme -s -q $qos -w $walltime --omp $omp -o $output_dir -p \$(param_args...)")
    
    if !dry_run
        original_dir = pwd()
        cd(CLIMBER_X_DIR)
        try
            cmd = `./runme -s -q $qos -w $walltime --omp $omp -o $output_dir -p $(param_args...)`
            println("\nActual command: $cmd")
            output = read(cmd, String)
            println("Output: $output")
            return output
        catch e
            println("Error: $e")
            return nothing
        finally
            cd(original_dir)
        end
    end
    
    return "Would run with $(length(param_args)) parameters"
end

"""
Method 4: Write a temporary bash script and execute it
"""
function test_method_bash_script(output_dir; dry_run=true, qos="standby", walltime="01:00:00", omp=32)
    println("\n" * "-"^60)
    println("METHOD 4: Temporary bash script")
    println("-"^60)
    
    # Create script content
    param_lines = ["           $(k)=$(v) \\" for (k, v) in TEST_PARAMS]
    # Remove trailing backslash from last line
    param_lines[end] = rstrip(param_lines[end], ['\\', ' '])
    
    script_content = """#!/bin/bash
cd $(CLIMBER_X_DIR)

./runme -s -q $(qos) -w $(walltime) --omp $(omp) \\
        -o "$(output_dir)" \\
        -p $(join(param_lines, "\n"))
"""
    
    println("\nGenerated script:")
    println("-"^40)
    println(script_content)
    println("-"^40)
    
    if !dry_run
        # Write to temp file
        script_file = tempname() * ".sh"
        open(script_file, "w") do f
            write(f, script_content)
        end
        chmod(script_file, 0o755)
        
        try
            println("\nExecuting $script_file...")
            output = read(`bash $script_file`, String)
            println("Output: $output")
            return output
        catch e
            println("Error: $e")
            return nothing
        finally
            rm(script_file)
        end
    end
    
    return script_content
end

"""
Run all diagnostic tests
"""
function run_diagnostics(; output_dir="/p/tmp/karinako/eki_test/diag_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
    println("="^60)
    println("CLIMBER-X SUBMISSION DIAGNOSTICS")
    println("="^60)
    println("Timestamp: $(now())")
    println("Test output directory: $output_dir")
    println("="^60)
    
    # Prerequisites
    if !check_prerequisites()
        println("\n✗ Prerequisites not met. Fix issues above first.")
        return
    end
    
    # Test each method (dry run first)
    println("\n" * "="^60)
    println("TESTING COMMAND CONSTRUCTION (DRY RUN)")
    println("="^60)
    
    test_method_bash_c(output_dir, dry_run=true)
    test_method_cmd_array(output_dir, dry_run=true)
    test_method_backtick(output_dir, dry_run=true)
    test_method_bash_script(output_dir, dry_run=true)
    
    println("\n" * "="^60)
    println("DIAGNOSTICS COMPLETE (dry run)")
    println("="^60)
    println("\nTo actually submit test jobs, run:")
    println("  submit_test_job(1)  # Method 1: bash -c")
    println("  submit_test_job(2)  # Method 2: Cmd array")
    println("  submit_test_job(3)  # Method 3: backtick")
    println("  submit_test_job(4)  # Method 4: temp script")
end

"""
Submit a test job using specified method
"""
function submit_test_job(method::Int; output_dir="/p/tmp/karinako/eki_test/method_$(method)_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
    println("Submitting test job using Method $method")
    println("Output will go to: $output_dir")
    
    result = if method == 1
        test_method_bash_c(output_dir, dry_run=false, qos="standby", walltime="20:00:00")
    elseif method == 2
        test_method_cmd_array(output_dir, dry_run=false, qos="standby", walltime="20:00:00")
    elseif method == 3
        test_method_backtick(output_dir, dry_run=false, qos="standby", walltime="20:00:00")
    elseif method == 4
        test_method_bash_script(output_dir, dry_run=false, qos="standby", walltime="20:00:00")
    else
        println("Unknown method: $method")
        return nothing
    end
    
    if result !== nothing
        # Try to extract job ID
        m = match(r"Submitted batch job (\d+)", result)
        if m !== nothing
            job_id = m.captures[1]
            println("\n✓ Job submitted! ID: $job_id")
            println("  Check status: squeue -j $job_id")
            println("  Cancel: scancel $job_id")
            return job_id
        end
    end
    
    return result
end

"""
Compare the Julia-generated command with your working bash command
"""
function compare_with_bash_script()
    println("="^60)
    println("COMPARISON WITH WORKING BASH SCRIPT")
    println("="^60)
    
    println("\nYour working bash script uses this pattern:")
    println("-"^60)
    working_bash = """
./runme -rs -q long -w 400:00:00 --omp 32 \\
    -o "\$run_dir" \\
    -p ocn.diff_dia_min="\$diff_dia_min" \\
       ocn.drag_topo_fac="\$drag_topo_fac" \\
       ocn.slope_max="\$slope_max" \\
       ocn.diff_iso="\$diff_iso" \\
       ocn.diff_gm="\$diff_gm" \\
       ocn.diff_dia_max="\$diff_dia_max" \\
       ctl.nyears=150000 \\
       ctl.co2_const=190 \\
       ctl.fake_geo_const_file=input/geo_ice_tarasov_12ka.nc \\
       ctl.fake_ice_const_file=input/geo_ice_tarasov_12ka.nc \\
       ctl.restart_in_dir="/home/karinako/climber-x/output/DO/spinup_ensemble/CO2_190/restart_out/year_3000" \\
       ocn.l_noise_fw=T \\
       ocn.noise_amp_fw=0.4 &
"""
    println(working_bash)
    
    println("\nKey differences to check:")
    println("  1. -rs vs -s flag (restart vs submit-only)")
    println("  2. Queue: Using standby")
    println("  3. Walltime: 20:00:00 for 10000 years")
    println("  4. Parameter: ocn.noise_amp_fw included")
    println("  5. Background (&) - bash runs async, Julia blocks")
end

# ============================================
# MAIN
# ============================================

println("CLIMBER-X Submission Diagnostics loaded!")
println("")
println("Available functions:")
println("  run_diagnostics()           - Run all diagnostic tests (dry run)")
println("  submit_test_job(method)     - Submit test job using method 1-4")
println("  compare_with_bash_script()  - Compare with your working bash")
println("  check_prerequisites()       - Check system requirements")
println("")
println("Start with: run_diagnostics()")