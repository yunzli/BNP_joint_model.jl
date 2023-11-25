function ij2h(i, j, Nvec)
    if j > Nvec[i]
        Error("Wrong index") 
    end 

    if i == 1 
        return j 
    end
    h = sum(Nvec[1:(i-1)]) + j
    return h
end

function h2ij(h, Nvec)
    i = 1 
    j = 0
    for N in Nvec
        if h > N 
            h -= N 
            i += 1
        else 
            j = h 
            return i, j
        end
    end 
end

function svd2inv(M)

	X = svd(M)
	Minv = X.Vt' * Diagonal(1 ./ X.S) * X.U'
	Minv = (Minv + Minv')/2

	return Minv 
end