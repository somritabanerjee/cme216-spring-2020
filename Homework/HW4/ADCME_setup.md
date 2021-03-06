---
layout: full-width
title: Installation Guide For ADCME 
---

[Link back to Homework 4](./HW4 Questions.md)

ADCME is tested and supported on Linux, macOS, and Windows (beta) systems. We have separate instructions for each operating systems.

![](assets/support_matrix.png){:width="40%"}

You can also use the Stanford [Farmshare](https://srcc.stanford.edu/farmshare2) computing environment. ssh to `rice.stanford.edu` using your SUNetID. Use the Linux installation guide for this.

Please see the first three videos on [Canvas](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=9a0acfdc-5022-4f2d-920d-abb90058a233) for a step-by-step installation guide.

# ADCME installation instructions

If you have some earlier version of Julia already installed with a version at least 1.0, it should work and you can skip the steps below. Go straight to step 2 "Install Project Dependencies." For Windows platforms, you will also need Microsoft Visual Studio 2017 15 (see below for instructions).

## 1. Install Julia

We will first install Julia binary and then configure the binary path so you can have easy access to Julia by typing `julia` in a terminal. But strictly speaking, configuring the path is unnecessary if your ADCME version $\geq 0.5.3$. 

### For Linux 

[Video with step-by-step instructions](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=f823ee8e-42b5-4004-a075-abb7005aa3b8)

See below for instructions on _rice_.

Download Julia 1.3 or 1.4 from the [official website](https://julialang.org/downloads/). Uncompress the tarball to any directory you want. There is a directory `bin` inside the Julia directory you just uncompressed. Add the absolute path of the `bin` directory to your `PATH` environment variable. 

Suppose the Julia `bin` path is `<LocalJuliaPath>` (e.g., `~/julia-1.4.1/bin`), execute the following command in your terminal:

```bash
echo 'export PATH=<LocalJuliaPath>:$PATH' >> ~/.bashrc
```

In case you use another shell (e.g., `zsh`) other than bash, you need to replace `~/.bashrc` in the command with the corresponding startup file. You can use `echo $SHELL` to check which shell you are using. 

---

For _rice_, Julia is already installed. Just run

```bash
$ module load julia/1.3.1
```

This will load Julia 1.3.1.

The installation is very slow on _rice_. Please be prepare to wait for a long time. The installation is on the order of one hour because the file system on _rice_ is very slow. Once everything is installed, running the code is relatively fast.

### For macOS

[Video with step-by-step instructions](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=027f5390-6cad-4190-ab91-abb70055cb3c)

Due to an incompatibility issue with Julia 1.4 and TensorFlow 1.x, please download and install Julia 1.3 from the [official website](https://julialang.org/downloads/oldreleases/#v131_dec_30_2019). 

After installation, Julia-1.3 will appear in your `Application` folder. Open the Julia application and you will see the Julia prompt

```julia
julia> Sys.BINDIR
```

Example output:

```bash
"/Applications/Julia-1.3.app/Contents/Resources/julia/bin"
```

Add this path to your `PATH` environment variable (make sure to scroll to the right to copy the entire line below)

```bash
echo 'export PATH=/Applications/Julia-1.3.app/Contents/Resources/julia/bin:$PATH' >> ~/.bash_profile
```

On the most recent version of macOS, you need to replace `~/.bash_profile` by `~/.zshrc`. If you are unsure, type `ls ~/.zshrc`. If the file exists, this is the one you should use.

### For Windows

[Video with step-by-step instructions](https://stanford-pilot.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=e17b2799-0590-405a-9536-abc20017a4d8)

If you have a Windows OS, you will need to install Microsoft Visual Studio 2017 15.

[Download and installation instructions](https://visualstudio.microsoft.com/vs/older-downloads/)

![](assets/vs2017.png){:width="60%"}

A free community version is available. Note that this is an older version of Visual Studio. It's not the one from 2019 but the previous version from 2017.

Then you can install Julia following these [instructions](https://julialang.org/downloads/). Choose your version of Windows (32-bit or 64-bit).

[Detailed instructions to install Julia on Windows](https://julialang.org/downloads/platform/#windows)

---

For Mac and Linux users, restart your shell to apply the new settings. Type `julia` in your terminal and you should see a Julia prompt (Julia REPL).

![](./assets/julia_prompt.png){:width="40%"}

---

For Windows users, you can press the Windows button or click the Windows icon (usually located in the lower left of your screen) and type `julia`. Open the Desktop App `Julia` and you will see a Julia prompt. 

![](./assets/windows.png){:width="40%"}

## 2. Install the Project Dependencies

This homework requires installing some Julia packages. Start julia 

```bash
$ julia
```

and type

```julia
julia> using Pkg
julia> Pkg.add("ADCME")
julia> Pkg.add("DelimitedFiles")
julia> Pkg.add("Conda")
julia> Pkg.add("PyCall")
julia> Pkg.add("PyPlot")
julia> Pkg.build("PyPlot") # if you have ADCME≧0.5.3, this step is not necessary
```

## 3. Start using ADCME

Now you can start using ADCME (ignore the warnings; they will disappear next time)

```julia
julia> using ADCME
julia> a = constant(ones(5,5))
julia> b = a * ones(5)
julia> sess = Session(); init(sess)
julia> run(sess, b)
```

Expected output:

```bash
5-element Array{Float64,1}:
 5.0
 5.0
 5.0
 5.0
 5.0
```

## 4. Test the Custom Operator Support

In the homework, we will use custom operators. To test whether your installation works for custom operators, try

```julia
julia> using ADCME
julia> ADCME.precompile()
```

If you encounter any compilation issue, you can report in Slack channel. 

### Compile the Custom Operator for 2D Case

The final step explains how to compile the custom operator for the 2D Case. 

In `2DCase`, you have two source files: `HeatEquation.h` and `HeatEquation.cpp`. You need to compile them into a shared library, which you can use in the inverse modeling. To do so, go into `2DCase` directory and open a Julia prompt in a terminal. 

```julia
julia> using ADCME
julia> mkdir("build")
julia> cd("build")
julia> ADCME.cmake()
julia> ADCME.make()
```

The command [ADCME.cmake()](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) will run commands in the file `CMakeLists.txt` and create the appropriate Makefile. Then the command [ADCME.make()](https://www.gnu.org/software/make/manual/make.html) will compile the source code `HeatEquation.h` and `HeatEquation.cpp` to create the shared library.

After running this, you should see that there is a `libHeatEquation.so` (Linux), `libHeatEquation.dylib` (macOS), or `HeatEquation.dll` (Windows) in your `build` directory. 

Run the `Case2D/example.jl` file to check if the shared library works. You may see some warning messages. If you see the following output at the end:

```shell
run(sess, err) = 2.9971950130484027e-6
Congratulations! `example.jl` completed successfully
```

the code ran successfully.

If you run the code within the Julia REPL, you will see a figure. For this, start julia in the directory `Case2D`, which contains `example.jl`:

```shell
$ cd Case2D
$ julia
```

Then type:

```shell
julia> include("example.jl")
```
 
You will see the same output as above `run(sess, err) = 2.9971950130484027e-6` along with this figure:

![](assets/example_output.png){:width="40%"}

You can rotate the figure in 3D using your mouse.

On _rice_, you will not see the figure because you are connected remotely through `ssh` and you cannot see graphic windows but you won't need this for the homework.

## Troubleshooting

On some Mac systems, you may encounter the following warning when you run `using PyPlot`.

---

PyPlot is using tkagg backend, which is known to cause crashes on macOS (#410); use the MPLBACKEND environment variable to request a different backend.

---

To fix this problem, run the following commands at a Julia prompt:

```julia
julia> using Conda
julia> Conda.add("pyqt") # if you install ADCME v0.5.2, this should be Conda.add("pyqt", :ADCME)
julia> using Pkg
julia> Pkg.build("PyPlot")
```