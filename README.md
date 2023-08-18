# SLP (Shiny Light Picker)
Development of a fast plate solver utilising geometric hashing and K-D Tree matching. This code is Python and designed for use on a CPU. 

The aim of the platesolver to rapidly produce a verified World Coordinate System (WCS) object for a query astronomical image.

The system is designed with navigation in mind and not as a general purpose plate solver such as astrometry.net

Solver steps:
1. Produce a catalogue. This is stored as a KD-Tree of hashcodes, a list of quads (4x star index) and a database of each catalogue star properties.
2. Take query image and process it into the same format as the catalogue i.e. Index, Tree, Quad List
3. Use a K-D Tree search to find the closest matching hashcodes from the query image and the catalogue. Produce hypothesis in the form of a WCS object.
4. Verfiy hypothesis by checking if nearby catalogue stars are represented in the image
5. Output the verfied WCS object
