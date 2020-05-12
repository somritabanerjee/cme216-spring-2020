using SparseArrays
using PyPlot 

Sys.isapple() && matplotlib.use("macosx")

n = 50
m = 50
U = zeros(n, m+1)

Δt = 1/m
Δx = 1/n 
x = Array((0:n)*Δx)
a = 5.0
b = 2.0
# κ = a .+ b * x 

κ = @. a + 1/(1+x^2) + x^2

λ = κ * Δt/Δx^2
F = ones(n, m) * Δt

λ_up = -λ[1:n-1]
λ_up[1] *= 2
A = spdiagm(
    -1 => -λ[2:n],
    0=>2λ[1:n] .+ 1,
    1 => λ_up
)

for k = 1:m 
    U[:, k+1] = A\(U[:,k] + F[:,k])
end 

close("all")
plot(U[1,:])
savefig("temperature.png")

DNN_LEFT_SIDE = Array(U[1:div(n,2), :]')