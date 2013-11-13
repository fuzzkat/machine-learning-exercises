s = 0
for i = 1:m
	x = X(i,:)
	h = sum(theta' .* x)
	s += (h - y(i))^2
endfor
J = 1/(2*m) * s
