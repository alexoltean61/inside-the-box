import sys
from skimage import io
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from skimage.transform import rescale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regression import SegmentedLeastSquares

class InsideTheBox:
	def __init__(self, max_len=1700, x_punish=5e-4, y_punish=1e-4, treshold=1e-3, look_back=25):
		self.MAX_LEN     = max_len
		self.X_PUNISH    = x_punish
		self.Y_PUNISH    = y_punish
		self.SLOPE_TREHSOLD = treshold
		self.LOOK_BACK = look_back
		self.sls = SegmentedLeastSquares(self.MAX_LEN)

	# axis: 0=oy, 1=ox
	def get_coord(self, x, y, axis):
		if axis == 0:
			return y
		return x

	def smoothen(self, graph, window):
		graph = pd.Series(graph)
		graph = graph.ewm(span=window).mean()
		return np.array(graph)

	def scale(self, graph):
		max = np.max(graph)
		min = np.min(graph)
		graph = (graph - min) / (max - min)
		return graph

	def read_image(self, filename):
		original = io.imread(filename)
		grayscale = rgb2gray(rgba2rgb(original))
		grayscale = grayscale > threshold_otsu(grayscale)
		return original, grayscale

	# axis: 0=oy, 1=ox
	def get_lines(self, table, axis=1):
		length = table.shape[axis]
		factor = 1
		if length > self.MAX_LEN:
			factor = self.MAX_LEN / length
			table = rescale(table, factor)

		y_max, x_max = table.shape
		max = self.get_coord(x_max, y_max, axis)
		frequency  = np.zeros(max)
		for x in range(x_max):
			for y in range(y_max):
				# todo: ignore black lines
				if table[y][x] == False:
					frequency[self.get_coord(x, y, axis)] += 1

		window = max // self.LOOK_BACK
		graph = self.scale(self.smoothen(frequency, window))
		graph = graph[window//2:]
		linears = self.sls.compute(np.arange(graph.shape[0]), graph, self.get_coord(self.X_PUNISH, self.Y_PUNISH, axis))
		#plt.plot(np.arange(graph.shape[0]), graph)
		lines = []
		downward = True
		for i, j, a, b in linears:
			#plt.plot(range(i,j+1), self.sls.linear(a, b, np.arange(i,j+1)))
			if downward == True and a >= self.SLOPE_TREHSOLD:
				lines.append(int(i/factor))
				#plt.axvline(int(i/factor))
				downward = False
			elif downward == False and a <= -self.SLOPE_TREHSOLD:
				downward = True
		#plt.show()
		return lines

	def get_boxes(self, table):
		column_points = self.get_lines(table, axis=1)
		row_points    = self.get_lines(table, axis=0)
		return (row_points,column_points)

def main():
	if len(sys.argv) != 2:
		print("Usage: python main.py [table_image_filename]")
		return
	filename = sys.argv[1]
	box_finder = InsideTheBox()
	orig, gray = box_finder.read_image("table3.png")
	row_points, column_points  = box_finder.get_boxes(gray)
	print(f"Columns: oX = {column_points}")
	print(f"Rows:    oY = {row_points}")
	orig[:,column_points,0:3] = 0
	orig[row_points,:,0:3] = 0
	io.imsave(filename+"_processed.png", orig)
	print(f"Saved! Check {filename+'_processed.png'}")


if __name__ == "__main__":
	main()