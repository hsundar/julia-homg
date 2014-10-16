module Tensor


function IAX(A::Array, x::Array)
    N = size(A, 1)
    y = A * reshape(x, N, N)
    y = reshape(y,size(y)[1].*size(y)[2],1)
end

function AIX(A::Array, x::Array)
    N = size(A, 1)
    y = A * reshape(x, N, N)'
    y = y'
    y = reshape(y,size(y)[1].*size(y)[2],1)
end

function IIAX(A::Array, x::Array)
    N = size(A, 1)
    y = A * reshape(x, N, N*N)
    y = reshape(y,size(y)[1].*size(y)[2],1)
end
function IAIX(A::Array, x::Array)
    N = size(A, 1)
    q = reshape(x, N, N, N)
    y = zeros(N,N,N)
    for i=1:N
        y[i,:,:] = A * squeeze(q[i,:,:])
    end
    y = reshape(y,size(y)[1].*size(y)[2],1)
end
function AIIX(A::Array, x::Array)
    N = size (A, 1)
    y = reshape(x, N*N, N) * A'
    y = reshape(y,size(y)[1].*size(y)[2],1)
end
function grad(refel, u)
    du = zeros(length(u), refel.dim);
    if (refel.dim == 2)
        du[:,1] = IAX(refel.Dr, u);
        du[:,2] = AIX(refel.Dr, u);
    else
        du[:,1] = IIAX(refel.Dr, u);
        du[:,2] = IAIX(refel.Dr, u);
        du[:,3] = AIIX(refel.Dr, u);
    end
    return du
end    
function Grad2(A::Array, x::Array)
    this.getIAX(A, x), this.getAIX(A, x)
end
function Grad3(A::Array, x::Array)
    this.getIIAX(A, x), this.getIAIX(A, x), this.getAIIX(A, x)
end

end

