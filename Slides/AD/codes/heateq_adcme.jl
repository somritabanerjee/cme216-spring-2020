using SparseArrays
using PyPlot 
using ADCME

Sys.isapple() && matplotlib.use("macosx")

n = 50
m = 50
U = zeros(n, m+1)

Δt = 1/m
Δx = 1/n 
x = Array((0:n)*Δx)
a = Variable(5.0)
b = Variable(2.0)
κ = a + b * x 

λ = κ * Δt/Δx^2
F = constant(ones(m, n) * Δt)

λ_up = -λ[1:n-1]
λ_up = scatter_update(λ_up, 1, 2λ_up[1])  # "equivalent" to λ_up[1] *= 2

A = spdiag(
    n,
    -1 => -λ[2:n],
    0=>2λ[1:n] + 1,
    1 => λ_up
)


function condition(k, U)
    k <= m 
end

function body(k, U)
    uk = read(U, k)
    Fk = F[k]
    rhs = uk + Fk 
    u_new = A\rhs
    U = write(U, k+1, u_new)
    k+1, U  
end

k = constant(1, dtype=Int32)
U = TensorArray(m+1)
U = write(U, 1, zeros(n))

_, U_out = while_loop(condition, body, [k, U])
U_array = set_shape(stack(U_out), (m+1, n))

left_side = U_array[:,1]

sess = Session(); init(sess)
LEFT_SIDE = run(sess, left_side)

close("all")
plot(LEFT_SIDE)
savefig("temperature_adcme.png")


