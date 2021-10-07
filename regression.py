import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, cpu_count

class SegmentedLeastSquares:
	def __init__(self, max_size, pool=None):
		self.linear = lambda a, b, x: a * x + b
		self.dotted_sums = np.zeros((max_size,max_size))
		self.x_sums      = np.zeros((max_size,max_size))
		self.y_sums      = np.zeros((max_size,max_size))
		self.x_sq_sums   = np.zeros((max_size,max_size))
		self.y_sq_sums   = np.zeros((max_size,max_size))
		self.errors = np.zeros((max_size-1,max_size))
		self.coeffs = [[(0,0) for _ in range(max_size)] for _ in range(max_size)]
		self.optim  = np.zeros(max_size)

	def regression(self, s1, s2, s3, s4, n):
		a = (n*s1 - s2*s3) / (n*s4 - s2**2)
		b = (s3 - a*s2) / n
		return (a, b)

	def regression_worker(self, i, j, s1, x_mult_y, s2, x, s3, y, s4, x_sq, s5, y_sq, n):
		s1 += x_mult_y
		s2 += x
		s3 += y
		s4 += x_sq
		s5 += y_sq
		a, b = self.regression(s1, s2, s3, s4, n)
		err  = (s5 - 2*a*s1 - 2*b*s3 + a**2*s4 + 2*a*b*s2 + n*(b**2)) / n
		return (a, b), err, s1, s2, s3, s4, s5

	def compute(self, x, y, C):
		n = np.shape(x)[0]

		x_mult_y = x * y
		x_sq     = x ** 2
		y_sq     = y ** 2
		np.fill_diagonal(self.dotted_sums, x_mult_y)
		np.fill_diagonal(self.x_sums, x)
		np.fill_diagonal(self.y_sums, y)
		np.fill_diagonal(self.x_sq_sums, x_sq)
		np.fill_diagonal(self.y_sq_sums, y_sq)

		for delta in range(1,n):
			for i in range(n-delta):
				j = i + delta
				self.coeffs[i][j], self.errors[i][j], self.dotted_sums[i][j], self.x_sums[i][j], self.y_sums[i][j], self.x_sq_sums[i][j], self.y_sq_sums[i][j] = self.regression_worker(i, j
					,self.dotted_sums[i][j-1], x_mult_y[j], self.x_sums[i][j-1],x[j]
					,self.y_sums[i][j-1], y[j], self.x_sq_sums[i][j-1], x_sq[j], self.y_sq_sums[i][j-1], y_sq[j], j-i+1)

		prev  = [0]
		for j in range(1, n):
			candidates = self.errors[:j,j] + self.optim[:j] + C
			i_min = np.argmin(candidates)
			prev.append(i_min)
			self.optim[j] = candidates[i_min]

		i = n-1
		ret = []
		while i > 0:
			start = prev[i]
			a, b = self.coeffs[start][i]
			ret.append((start, i, a, b))
			i = start
		return ret[::-1]

def main():
	sls = SegmentedLeastSquares(1400)
	size = 800
	x    = np.linspace(start=342, stop=34523, num=size)
	y1    = sls.linear(333, 123, x[:size//2]) + 33*np.random.normal(size=size//2)
	y2    = sls.linear(195, -666, x[size//2:]) + 10*np.random.normal(size=size//2)
	y     = np.concatenate((y1, y2))#linear(1, 0, x + np.random.normal(size=size)) #
	del sls

if __name__ == "__main__":
	main()