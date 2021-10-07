import sys
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import threshold_otsu
from skimage.transform import rescale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from regression import SegmentedLeastSquares

class InsideTheBox:
	def __init__(self, max_len=1400, x_punish=5e-4, y_punish=3e-4, treshold=1e-3, look_back=20, black=3e-1):
		self.MAX_LEN     = max_len
		self.X_PUNISH    = x_punish
		self.Y_PUNISH    = y_punish
		self.SLOPE_TREHSOLD = treshold
		self.LOOK_BACK = look_back
		self.BLACK     = black
		self.LIMIT     = 0.5
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
		grayscale, stripes = self.remove_stripes(grayscale)
		return original, grayscale, stripes

	def remove_stripes(self, table):
		y_max, x_max = table.shape
		unchanged = copy.copy(table)
		column_points = []
		row_points    = []
		for y in range(y_max):
			if np.mean(unchanged[y]) < self.BLACK:
				table[max(0,y-5):min(y_max,y+5),:] = True
				row_points.append(y)
		for x in range(x_max):
			if np.mean(unchanged[:,x]) < self.BLACK:
				table[:,max(0,x-5):min(x_max,x+5)] = True
				column_points.append(x)
		return table, (column_points, row_points)

	# axis: 0=oy, 1=ox
	def get_lines(self, table, axis=1):
		length = table.shape[axis]
		factor = 1
		modulus = 0
		if length > self.MAX_LEN:
			factor = self.MAX_LEN / length
			modulus = length % self.MAX_LEN
			table = rescale(table, factor)
		punish = table.shape[axis] / self.get_coord(self.X_PUNISH, self.Y_PUNISH, axis)
		#print(punish)

		y_max, x_max = table.shape
		c_max = self.get_coord(x_max, y_max, axis)
		frequency  = np.zeros(c_max)
		for x in range(x_max):
			for y in range(y_max):
				# todo: ignore black lines
				if table[y][x] == False:
					frequency[self.get_coord(x, y, axis)] += 1

		window = c_max // self.LOOK_BACK
		graph = self.scale(self.smoothen(frequency, window))
		graph[graph > self.LIMIT] = self.LIMIT
		graph = self.scale(graph)

		linears = self.sls.compute(np.arange(graph.shape[0]), graph, punish)

		plt.plot(range(graph.shape[0]), graph)
		lines = []
		downward = True
		last = 100
		for i, j, a, b in linears:
			curr = self.sls.linear(a, b, i)
			last2= self.sls.linear(a, b, j)
			plt.plot(range(i,j+1), self.sls.linear(a, b, np.arange(i, j+1)))
			lgth = np.linalg.norm(curr-last2)
			if lgth >= 0.1:
				if downward == True and (a >= self.SLOPE_TREHSOLD or curr-last>=7e-2):
					plt.axvline(i)
					lines.append(int(i/factor))
					downward = False
				elif downward == False and (a <= -self.SLOPE_TREHSOLD or last-curr>=7e-2):
					downward = True
				last = last2

		return lines

	def get_boxes(self, table, filename):
		plt.clf()
		column_points = self.get_lines(table, axis=1)
		plt.savefig(filename + "_0columns.png", dpi=300)
		plt.clf()
		row_points    = self.get_lines(table, axis=0)
		plt.savefig(filename + "_0rows.png", dpi=300)
		return (row_points,column_points)

def main():
	if len(sys.argv) < 2:
		print("Usage: python main.py [table_image_filename]")
		return
	for filename in sys.argv[1:]:
		box_finder = InsideTheBox(x_punish=21e5,y_punish=1e6, look_back=20)
		orig, gray, stripes = box_finder.read_image(filename)
		print(f"Original columns: oX = {stripes[0]}")
		print(f"Original rows:    oY = {stripes[1]}")
		row_points, column_points  = box_finder.get_boxes(gray, filename)
		print(f"Columns: oX = {column_points}")
		print(f"Rows:    oY = {row_points}")
		gray[:,column_points] = False
		gray[row_points,:] = False
		io.imsave(filename+"_processed.png", img_as_ubyte(gray))
		print(f"Saved! Check {filename+'_processed.png'}")

if __name__ == "__main__":
	main()