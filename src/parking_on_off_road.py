#!/usr/bin/env python3
import math
import optparse
import os
import sys
from contextlib import nullcontext
from xxlimited_35 import error


from sumolib import checkBinary
import traci
import xml.etree.ElementTree as ET
import traci.constants as tc
import sumolib
import numpy as np
from collections import defaultdict
import random
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy
import imageio.v2 as imageio
import shutil
import concurrent.futures


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


def get_parking_coordinates(parking_id, additional_file):
    """
    Get the coordinates of a parking area.

    Parameters:
    - parking_id: The ID of the parking area.
    - additional_file: The path to the additional file (additional.xml).

    Returns:
    - A tuple with the parking area coordinates (x, y) or None if not found.
    """
    import xml.etree.ElementTree as ET  # Import the module for XML handling

    try:
        # Parse the additional file to get parking area information
        tree_additional = ET.parse(additional_file)
        root_additional = tree_additional.getroot()

        # Find the specified parking area
        parking_area = root_additional.find(f".//parkingArea[@id='{parking_id}']")
        if parking_area is None:
            # print(f"ParkingArea with ID {parking_id} not found in {additional_file}")
            return None

        lane_id = parking_area.get("lane")
        start_pos = float(parking_area.get("startPos"))
        end_pos = float(parking_area.get("endPos"))

        # Debugging
        # (f"Found parking area: Lane ID: {lane_id}, Start Position: {start_pos}, End Position: {end_pos}")

        # Get the lane coordinates
        lane_coords = traci.lane.getShape(lane_id)
        if not lane_coords:
            # print(f"No coordinates found for lane {lane_id}")
            return None

        # print(f"Lane coordinates: {lane_coords}")

        # Calculate the parking area coordinates
        parking_coords = []

        for i in range(len(lane_coords) - 1):
            x1, y1 = lane_coords[i]
            x2, y2 = lane_coords[i + 1]
            segment_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Check if the parking area is within the segment
            if start_pos < (i + 1) * segment_length and end_pos > i * segment_length:
                if segment_length > 0:
                    proportion_start = (start_pos - i * segment_length) / segment_length
                    proportion_end = (end_pos - i * segment_length) / segment_length

                    x_start = x1 + (x2 - x1) * proportion_start
                    y_start = y1 + (y2 - y1) * proportion_start
                    x_end = x1 + (x2 - x1) * proportion_end
                    y_end = y1 + (y2 - y1) * proportion_end

                    # Add only the first calculated point
                    # print(f"aggiunto {x_start}")
                    parking_coords.append((x_start, y_start))
                    break  # Exit the loop after finding the first coordinate

        # Return the first found coordinate
        if parking_coords:
            # print("ritorna coordinate")
            return parking_coords[0]

    except Exception as e:
        print(f"An error occurred: {e}")

    # print(f"ritorna none per il parcheggio {parking_id}")
    return None  # Return None if the parking area is not found


class HeatMap:

    def __init__(self, xml_file, additional_file):
        """
        Initialize a new heatmap by reading the area size from an XML file
        and automatically calculating the map boundaries.

        Parameters:
        - xml_file: The path to the XML file from which to read 'area_size'.
        """
        # Read the area size from the XML file
        self.area_size = self._read_area_size_from_xml(xml_file)

        # Get the network boundaries automatically
        (self.minX, self.minY), (self.maxX, self.maxY) = traci.simulation.getNetBoundary()
        # print(f"Net Boundaries: minX={self.minX}, minY={self.minY}, maxX={self.maxX}, maxY={self.maxY}")

        # Adjust the boundaries to include all parking areas
        self._expand_boundaries_for_parking(additional_file)

        # Calculate the matrix dimensions
        self.cols = math.ceil((self.maxX - self.minX) / self.area_size)
        self.rows = math.ceil((self.maxY - self.minY) / self.area_size)

        # Initialize the heatmap matrix as a list of lists
        self.heat_map = [[[] for _ in range(self.cols)] for _ in range(self.rows)]

    def _expand_boundaries_for_parking(self, additional_file):
        """
        Expand the map boundaries to include all parking area positions.

        Parameters:
        - additional_file: The XML file containing parking area information.
        """
        tree_additional = ET.parse(additional_file)
        root_additional = tree_additional.getroot()

        # Find all 'parkingArea' elements
        for parking_area in root_additional.findall(".//parkingArea"):
            parking_id = parking_area.get("id")
            posX, posY = get_parking_coordinates(parking_id, additional_file)

            if posX is not None and posY is not None:
                # Expand the boundaries if necessary
                if posX < self.minX:
                    self.minX = posX
                if posX > self.maxX:
                    self.maxX = posX
                if posY < self.minY:
                    self.minY = posY
                if posY > self.maxY:
                    self.maxY = posY

    def _read_area_size_from_xml(self, xml_file):
        """
        Read the 'area_size' value from an XML file.

        Parameters:
        - xml_file: The path to the XML file.

        Returns:
        - area_size: The value read from the XML file.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find the 'area_size' element and return its value as a float
        area_size = float(root.find('area_size').text)
        return area_size

    def update(self, parkage, parked_id=None, real_parkages=False):
        # heatmap case
        if not real_parkages:

            """
            Update the heatmap based on the parking area position and parkage state.
            """

            posX, posY = get_parking_coordinates(parked_id, 'parking_on_off_road.add.xml')

            # Calculate the matrix indices for the vehicle position
            col_index = math.floor((posX - self.minX) / self.area_size)
            row_index = math.floor((posY - self.minY) / self.area_size)

            # Check if the indices are within the matrix limits
            if 0 <= col_index < self.cols and 0 <= row_index < self.rows:
                if not parkage:
                    self.heat_map[row_index][col_index].append(1)
                else:
                    self.heat_map[row_index][col_index].append(-1)
        # real parking map case
        else:
            """
            Find all parking areas in the road network.
            """
            tree_additional = ET.parse('parking_on_off_road.add.xml')
            root_additional = tree_additional.getroot()

            # Find all 'parkingArea' elements
            for parking_area in root_additional.findall(".//parkingArea"):
                parking_id = parking_area.get("id")
                roadside_capacity = int(parking_area.get("roadsideCapacity"))  # Get the parking area capacity
                posX, posY = get_parking_coordinates(parking_id, 'parking_on_off_road.add.xml')
                # Calculate the matrix indices for the vehicle position
                col_index = math.floor((posX - self.minX) / self.area_size)
                row_index = math.floor((posY - self.minY) / self.area_size)

                # Check if the indices are within the matrix limits
                if 0 <= col_index < self.cols and 0 <= row_index < self.rows:
                    self.heat_map[row_index][col_index].append(roadside_capacity)

    def print_heatmap(self, title="Heatmap", real_parkage=False):
        """
        Print the heatmap using Matplotlib, coloring only cells with non-empty lists.

        Parameters:
        - title: The heatmap title (optional).
        """
        # Create a matrix for visualization
        display_matrix = np.zeros((self.rows, self.cols))  # Use 0 for empty cells

        # Assign the value 1 to cells with a non-empty list
        for i in range(self.rows):
            for j in range(self.cols):
                if self.heat_map[i][j]:  # Check if the list is not empty
                    display_matrix[i, j] = 1

        # Define the colormap: one color for cells with data, one for cells without data
        cmap = mcolors.ListedColormap(['white', 'blue'])  # 'white' for cells without data, 'blue' for cells with data

        # Define the colormap boundaries
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(10, 8))
        cax = plt.imshow(display_matrix, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')

        # Add a color bar for the heatmap
        cbar = plt.colorbar(cax, ticks=[0, 1])

        if not real_parkage:
            cbar.set_label('Vehicle presence')
        else:
            cbar.set_label('Parking area presence')
        cbar.set_ticks([0, 1])

        if not real_parkage:
            cbar.set_ticklabels(['No parking', 'Parking'])

        # Add a border to the grid boundaries
        for i in range(self.rows + 1):
            plt.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.cols + 1):
            plt.axvline(j - 0.5, color='black', linewidth=1)

        plt.title(title)
        plt.xlabel('Grid column')
        plt.ylabel('Grid row')
        plt.show()

    def get_heatmap(self):
        """
        Return the heatmap matrix.
        """
        return self.heat_map

    def print_heatmap_values(self):
        """
        Print the values inside each cell of the heatmap.
        """
        print("Heatmap Values:")
        for i in range(self.rows - 1, -1, -1):  # Start from the last row and go to the first
            row_values = []
            for j in range(self.cols):
                cell_value = sum(self.heat_map[i][j])
                row_values.append(f"{cell_value:4d}")
            print(" ".join(row_values))

        """for i in range(self.rows):
            for j in range(self.cols):
                print(f"elemento heatmap[{i}][{j}] : {self.heat_map[i][j]}")
        """

    def save_heatmap_to_image(self, file_path, title="Heatmap", real_parkage=False):
        """
        Save the heatmap as an image file.

        Parameters:
        - file_path: The path to the image file where to save the heatmap.
        - title: The heatmap title (optional).
        """
        # Create a matrix for visualization
        display_matrix = np.zeros((self.rows, self.cols))  # Use 0 for empty cells

        # Assign the value 1 to cells with a non-empty list
        for i in range(self.rows):
            for j in range(self.cols):
                if self.heat_map[i][j]:  # Check if the list is not empty
                    display_matrix[i, j] = 1

        # Define the colormap: one color for cells with data, one for cells without data
        cmap = mcolors.ListedColormap(['white', 'blue'])  # 'white' for cells without data, 'blue' for cells with data

        # Define the colormap boundaries
        bounds = [-0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(10, 8))
        cax = plt.imshow(display_matrix, cmap=cmap, norm=norm, interpolation='nearest', origin='lower')

        # Add a color bar for the heatmap
        if not real_parkage:
            cbar = plt.colorbar(cax, ticks=[0, 1])
            cbar.set_label('Vehicle presence')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['No Data', 'Data'])
        else:
            cbar = plt.colorbar(cax, ticks=['No', 'Yes'])
            cbar.set_label('Parking area presence')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['No Parking', 'Parking'])

        # Add a border to the grid boundaries
        for i in range(self.rows + 1):
            plt.axhline(i - 0.5, color='black', linewidth=1)
        for j in range(self.cols + 1):
            plt.axvline(j - 0.5, color='black', linewidth=1)

        plt.title(title)
        plt.xlabel('Grid column')
        plt.ylabel('Grid row')

        # Save the plot as an image
        plt.savefig(file_path, bbox_inches='tight')

        # Close the figure to free memory
        plt.close()

        # print(f"Heatmap saved as image in {file_path}")

    def get_coordinates_from_cell(self, row, col):
        """
        Calculate the central (X, Y) coordinates of the given cell in the heatmap matrix.
        """
        x = self.minX + col * self.area_size + self.area_size / 2
        y = self.minY + row * self.area_size + self.area_size / 2
        return x, y

    def find_closest_lane(self, posX, posY):
        """
        Find the closest lane to the specified coordinates using Euclidean distance.
        """
        lanes = traci.lane.getIDList()
        closest_lane = None
        min_distance = float('inf')

        for lane_id in lanes:
            lane_shape = traci.lane.getShape(lane_id)
            if not lane_shape:
                continue

            for i in range(len(lane_shape) - 1):
                (x1, y1) = lane_shape[i]
                (x2, y2) = lane_shape[i + 1]
                distance = self.distance_point_to_segment(posX, posY, x1, y1, x2, y2)
                if distance < min_distance:
                    min_distance = distance
                    closest_lane = lane_id

        return closest_lane

    def distance_point_to_segment(self, px, py, x1, y1, x2, y2):
        """
        Calculate the distance from the point (px, py) to the segment defined by (x1, y1) and (x2, y2).
        """
        segment_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if segment_length_squared == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / segment_length_squared))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

    """,preference"""

    """,preference"""

    def direct_vehicle_to_best_parking(self, vehicle_id, destinations, parkage_map, net, alfa):
        """
        Parameters:
        - vehicle_id: The ID of the vehicle to be directed to a parking area.
        - destinations: Dictionary that maps vehicle_id to edge_id, representing the current destination of the vehicle.
        """
        edge_id = destinations.get(vehicle_id)

        if edge_id is None:
            # print(f"No destination found for vehicle with ID {vehicle_id}.")
            return

        """
        We need to find a tradeoff between the distance of a parking area and the probability that it is actually free.
        To implement this, we consider the formula score = alfa * H(i) + (1-alfa) * D(i,d).
        alfa is the multiplicative coefficient (0.5 to give equal weight), i is the i-th cell in the heatmap representation,
        H is the norm of the probability of a parking area being free, while D is the norm of the distance of the parking area from point B, represented by d.
        """

        DIS_SUBMAP = 350         # norma D consideriamo una sottomappa di raggio 5 km

        max_score = -1000 #provare a settare a -2
        best_lane = None

        print(f'Possible new destinations for {vehicle_id}')

        for i in range(self.rows):
            for j in range(self.cols):
                if self.heat_map[i][j]:  # if marked on the heatmap
                    num_true_parkage = sum(parkage_map.heat_map[i][j])  # Total number of parking spots
                    occupied_parkage = sum(self.heat_map[i][j]) * (-1)  # Number of occupied parking spots
                    norm_parkage = 1.00 - (float(occupied_parkage) / num_true_parkage)  # Norm H

                    posX, posY = self.get_coordinates_from_cell(i, j)  # Central coordinates of the cell
                    nearest_lane = self.find_closest_lane(posX, posY)  # Lane closest to the coordinates
                    distance_to_B, _ = get_route_distance(net, edge_id, nearest_lane)

                    norm_distance = 1.00 - (float(distance_to_B) / DIS_SUBMAP)  # Norm D

                    score = alfa * norm_parkage + (1 - alfa) * norm_distance

                    if score > max_score:
                        max_score = score
                        best_lane = nearest_lane

                    print("--------------------------------------------------------------------------------")
                    print(f"alfa: {alfa}")
                    print(f"Lane {nearest_lane.split('_')[0]} score : {score}")
                    print(f"details - norm H : {norm_parkage} norm D : {norm_distance}")
                    print(f"total parking spots: {num_true_parkage} occupied parking spots: {occupied_parkage} ")
                    print(f"distance from B: {distance_to_B} ")
                    print(f"{alfa} * {norm_parkage} + (1-{alfa}) * {norm_distance} = {score}")
                    print("--------------------------------------------------------------------------------")

        if best_lane:
            traci.vehicle.changeTarget(vehicle_id, best_lane.split('_')[0])
            destinations[vehicle_id] = best_lane.split('_')[0]
            print(
                f"Vehicle {vehicle_id} has been directed to lane: {best_lane.split('_')[0]} with score {max_score}")
        else:
            print(f"No valid lane found for vehicle {vehicle_id}.")


def is_vehicle_parked(vehicle_id):
    stop_state = traci.vehicle.getStopState(vehicle_id)
    return stop_state & tc.STOP_PARKING != 0


def is_near_parkage(vehicle_id, parkage_id, parking_to_edge):
    if parking_to_edge[parkage_id] == traci.vehicle.getRoadID(vehicle_id).split('_')[0]:  # if on the same lane
        vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
        park_start = traci.parkingarea.getStartPos(parkage_id)
        park_end = traci.parkingarea.getEndPos(parkage_id)

        if vehicle_position > park_start - 15 and vehicle_position < park_end - 15:
            # print(f"vehicle {vehicle_id} near parking area {parkage_id}")
            return True

    return False


def park_vehicle(vehicle_id, parkage_id, parking_car_parked, parking_capacity, parked_vehicles):
    occupied_count = parking_car_parked[parkage_id]
    capacity = parking_capacity[parkage_id]

    if occupied_count < capacity:
        print("There is space")

        # Gradual braking before stopping the vehicle (new)
        target_speed = 2  # Set a low speed before parking, e.g., 2 m/s
        duration = 5  # Time to slow down (in seconds)
        traci.vehicle.slowDown(vehicle_id, target_speed, duration)


        time_stop = random.randint(300, 400)
        traci.vehicle.setParkingAreaStop(vehicle_id, parkage_id, time_stop)  # time_stop is the duration of the stop
        parking_car_parked[parkage_id] += 1
        parked_vehicles[vehicle_id] = parkage_id
        return True

    return False


def is_exit_Parkage(vehicle_id, parking_id, parking_to_edge):
    if parking_to_edge[parking_id] == traci.vehicle.getRoadID(vehicle_id).split('_')[0]:
        vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
        park_end = traci.parkingarea.getEndPos(parking_id)

        if vehicle_position > park_end:
            return True
    return False


def get_route_distance(net, from_edge, to_edge):
    from_edge_obj = net.getEdge(from_edge.split('_')[0])
    to_edge_obj = net.getEdge(to_edge.split('_')[0])
    route = net.getShortestPath(from_edge_obj, to_edge_obj)
    """if route:
        distance = sum(edge.getLength() for edge in route[0])
        return distance, route"""

    mid_point_B = get_midpoint(net.getEdge(from_edge.split('_')[0]))
    mid_point_parkage = get_midpoint(net.getEdge(to_edge.split('_')[0]))
    pedon_distance = calculate_distance(mid_point_B,
                                        mid_point_parkage)  # calculate the distance parking area - point B as

    return pedon_distance, None
    #return float('inf'), None


# in the future, we will consider the heatmap using Beacon
def find_empty_parkages(parking_capacity, parking_list):
    empty_parkages = []
    for p in parking_list:
        if len(traci.parkingarea.getVehicleIDs(p)) < parking_capacity[p]:
            empty_parkages.append(p)
    return empty_parkages


def is_vehicle_near_junction(vehID, net, threshold_distance=20.0):
    try:
        # Get the current position of the vehicle
        vehicle_position = traci.vehicle.getPosition(vehID)
        # print(f"Vehicle {vehID} position: {vehicle_position}")

        # Get the ID of the edge where the vehicle is located
        current_edge = traci.vehicle.getRoadID(vehID)
        # print(f"Edge {current_edge}")

        # Check if the current edge is a junction
        if current_edge.startswith(':'):
            # print(f"{current_edge} is a junction, not an edge.")
            return False, None

        # Get the ID of the destination junction of the current edge
        next_junction_id = net.getEdge(current_edge).getToNode().getID()

        # print(f"Vehicle {vehID} Next Junction ID: {next_junction_id}")

        if not next_junction_id:
            # print(f"No junction found for edge {current_edge}")
            return False, None

        # Get the position of the next junction
        junction_position = traci.junction.getPosition(next_junction_id)
        # print(f"Junction {next_junction_id} position: {junction_position}")

    except Exception as e:
        print(f"Error: {e}")
        return False, None

        # Calculate the distance between the vehicle and the next junction
    distance = np.linalg.norm(np.array(vehicle_position) - np.array(junction_position))
    # print(f"Distance to junction {next_junction_id}: {distance}")

    # Check if the vehicle is within the threshold distance from the next junction
    if distance <= threshold_distance:
        # print(f"Vehicle {vehID} near the next junction {next_junction_id}")
        return True, next_junction_id

    return False, None


def get_reachable_edges_from_lane(laneID):
    reachable_edges = []

    # Get the list of connections (next lanes) from the current lane
    connections = traci.lane.getLinks(laneID)

    for conn in connections:
        connected_laneID = conn[0]  # The connected lane is the first element of the tuple
        connected_edgeID = traci.lane.getEdgeID(connected_laneID)

        if connected_edgeID not in reachable_edges:
            reachable_edges.append(connected_edgeID)

        # it's the entry edge
        if 'E2' in reachable_edges:
            reachable_edges.remove('E2')

    return reachable_edges


def select_next_edge(vehicle_id, lane_history, possible_edges):
    """
    Select the next lane for the vehicle, giving priority to lanes with fewer passes.
    """
    # Get the number of passes for each of the possible lanes
    pass_counts = [lane_history[vehicle_id].get(edge, 0) for edge in possible_edges]

    # Calculate probabilities inversely proportional to the number of passes
    max_pass_count = max(pass_counts) if pass_counts else 1
    weights = [max_pass_count - count + 1 for count in pass_counts]  # More passes, less weight

    # Select a lane based on these probabilities
    selected_lane = random.choices(possible_edges, weights=weights, k=1)[0]
    return selected_lane, pass_counts


def set_vehicle_route(vehicle_id, edge_history, reachable_edges):
    current_time = traci.simulation.getTime()

    """
    Set the vehicle's route to one of the possible lanes, giving priority to lanes with fewer passes.
    """
    selected_edge, pass_count = select_next_edge(vehicle_id, edge_history, reachable_edges)

    # Set the vehicle's route to the selected edge
    traci.vehicle.changeTarget(vehicle_id, selected_edge)

    # Update the pass count on the selected lane
    if selected_edge not in edge_history[vehicle_id]:
        edge_history[vehicle_id][selected_edge] = 0
    edge_history[vehicle_id][selected_edge] += 1

    return selected_edge, current_time, pass_count


def extract_number(vehicle):
    # Split the string on the dot and take the part after the dot
    return int(vehicle.split('.')[1])


def random_point_on_map():
    # Get the map boundaries

    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    NET_FILE = "parking_on_off_road.net.xml"
    tree = ET.parse(NET_FILE)
    root = tree.getroot()
    ENTRY_EXIT_LIST = ["E85", "-E85", "E86", "-E86"]

    # calculate arrival lane
    try:
        # Iterate over all edges
        for edge in root.findall('edge'):
            # Iterate over all lanes within the edge
            if edge.get("id") not in ENTRY_EXIT_LIST:
                for lane in edge.findall('lane'):
                    # Get the value of the shape attribute
                    shape = lane.get('shape')
                    if shape:
                        # Parse the points of the shape attribute
                        points = shape.split()
                        for point in points:
                            x, y = map(float, point.split(','))
                            # Update the boundaries
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x)
                            max_y = max(max_y, y)




    except Exception as e:

        print(f"Error while getting map boundaries from network file: {e}")

        return None, None, None, None

    # Generate a random point within the map boundaries

    random_x = random.uniform(min_x, max_x)

    random_y = random.uniform(min_y, max_y)

    # Randomly generate edge_id = None with a 10% probability

    # if random.random() < 0.1:  # 10% probability

    #   return (random_x, random_y), None

    # Get the closest edge to the random point

    edge_id = None

    while edge_id == None or edge_id in ENTRY_EXIT_LIST:

        # print(f"edge id: {edge_id}")

        try:

            random_x = random.uniform(min_x, max_x)

            random_y = random.uniform(min_y, max_y)

            edge_id, _, _ = traci.simulation.convertRoad(random_x, random_y, isGeo=False)

        except traci.TraCIException:

            edge_id = None

    connected_edges = []

    # Check if the edge_id is a junction and find a connected edge

    if edge_id and edge_id.startswith(':'):

        for edge in root.findall('edge'):

            # print(f"to: {edge.get('to')}, current {(edge_id.split('_')[0])[1:]}")

            if edge.get("to") == (edge_id.split('_')[0])[1:]:
                connected_edges.append(edge.get("id"))

        for e in connected_edges:

            if not e.startswith(':') and e not in ENTRY_EXIT_LIST:
                edge_id = e

                break

    # calculate departure lane

    try:

        # Iterate over all edges

        for edge in root.findall('edge'):

            # Iterate over all lanes within the edge

            if edge.get("id") not in ENTRY_EXIT_LIST:

                for lane in edge.findall('lane'):

                    # Get the value of the shape attribute

                    shape = lane.get('shape')

                    if shape:

                        # Parse the points of the shape attribute

                        points = shape.split()

                        for point in points:
                            x, y = map(float, point.split(','))

                            # Update the boundaries

                            min_x = min(min_x, x)

                            min_y = min(min_y, y)

                            max_x = max(max_x, x)

                            max_y = max(max_y, y)


    except Exception as e:

        print(f"Error while getting map boundaries from network file: {e}")

        return None, None, None, None

        # Generate a random point within the map boundaries

    random_x = random.uniform(min_x, max_x)

    random_y = random.uniform(min_y, max_y)

    # Randomly generate edge_id = None with a 10% probability

    # if random.random() < 0.1:  # 10% probability

    #   return (random_x, random_y), None

    # Get the closest edge to the random point

    edge_id_p = None

    while edge_id_p == None or edge_id_p in ENTRY_EXIT_LIST:

        # print(f"edge id: {edge_id}")

        try:

            random_x = random.uniform(min_x, max_x)

            random_y = random.uniform(min_y, max_y)

            edge_id_p, _, _ = traci.simulation.convertRoad(random_x, random_y, isGeo=False)

        except traci.TraCIException:

            edge_id_p = None

    connected_edges = []

    # Check if the edge_id is a junction and find a connected edge

    if edge_id_p and edge_id_p.startswith(':'):

        for edge in root.findall('edge'):

            # print(f"to: {edge.get('to')}, current {(edge_id.split('_')[0])[1:]}")

            if edge.get("to") == (edge_id_p.split('_')[0])[1:]:
                connected_edges.append(edge.get("id"))

        for e in connected_edges:

            if not e.startswith(':') and e not in ENTRY_EXIT_LIST:
                edge_id_p = e

                break

    """,preference"""

    return (random_x, random_y), edge_id_p, edge_id


def get_vehicle_number_from_xml(xml_file):
    tree = ET.parse(xml_file)

    root = tree.getroot()

    vehicles_number = int(root.find('vehicles_number').text)

    return vehicles_number


def set_vehicle_point_A_B(vehicle_number):
    # CSV file name

    file_name = f'setDestination_{vehicle_number}.csv'

    # Check if the file already exists

    if os.path.exists(file_name):
        print(f"{file_name} already exists. It will not be overwritten.")

        return

    # Get the list of vehicles from the XML file

    # vehicle_number = get_vehicle_number_from_xml('heat_map.xml') #to modify

    vehicle_ids = []

    for i in range(vehicle_number):
        vehicle_ids.append(f"vehicle_{i}")

    # List to save vehicle destinations

    destinations = []

    for vehicle_id in vehicle_ids:

        # Get a random point on the map

        """, preference"""

        (x, y), edge_id_p, edge_id = random_point_on_map()

        if edge_id:

            """,preference"""

            destinations.append([vehicle_id, edge_id_p, edge_id, x, y])

        else:

            print(f"Error creating destination for vehicle {vehicle_id}")

            destinations.append([vehicle_id, "None"])

    # Write destinations to the CSV file

    with open(file_name, mode='w', newline='') as file:

        writer = csv.writer(file)

        """,'preference'"""

        writer.writerow(['VehicleID', 'EdgeIDp', 'EdgeID', 'X', 'Y'])

        for destination in destinations:
            writer.writerow(destination)

    print(f"{file_name} has been created with vehicle destinations.")


def get_vehicle_point_A_B(probability_heatmap, vehicle_number):
    # CSV file name

    file_name = f'setDestination_{vehicle_number}.csv'

    # Check if the file exists

    if not os.path.exists(file_name):
        print(f"{file_name} does not exist. Make sure the file exists and try again.")

        return

    # Read destinations from the CSV file

    starting_lanes = {}

    destinations = {}

    use_heatmap = {}

    # preference = {}

    with open(file_name, mode='r') as file:

        reader = csv.DictReader(file)

        for row in reader:

            vehicle_id = row['VehicleID']

            edge_id_p = row['EdgeIDp']

            edge_id = row['EdgeID']

            starting_lanes[vehicle_id] = edge_id_p

            destinations[vehicle_id] = edge_id

            random_float = random.uniform(0, 1)

            if random_float < probability_heatmap:

                use_heatmap[vehicle_id] = 'True'

            else:

                use_heatmap[vehicle_id] = 'False'

            # preference[vehicle_id] = row['preference']

    return starting_lanes, destinations, use_heatmap  # ,preference


def calculate_distance(point1, point2):
    """

    Calculate the Euclidean distance between two points.


    Args:

        point1: A tuple with the coordinates of the first point (x1, y1).

        point2: A tuple with the coordinates of the second point (x2, y2).


    Returns:

        The Euclidean distance between the two points.

    """

    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_midpoint(edge):
    """

    Calculate the midpoint of an edge (lane).


    Args:

        edge: Edge object, which has start and end coordinates.


    Returns:

        A tuple containing the coordinates (x, y) of the midpoint.

    """

    start = edge.getFromNode().getCoord()  # Start coordinates

    end = edge.getToNode().getCoord()  # End coordinates

    midpoint_x = (start[0] + end[0]) / 2

    midpoint_y = (start[1] + end[1]) / 2

    return (midpoint_x, midpoint_y)


def calculate_data_analysis(time_A_start, time_B_arrive, parked_in_B, vehicle_use_heatmap, time_parked, edge_parked,

                            origin_destinations, net):
    total_time_vehicle_heatmap = 0

    total_time_vehicle_not_heatmap = 0

    num_parking_b_heatmap = 0

    num_parking_b_not_heatmap = 0

    num_use_heatmap = 0

    num_not_use_heatmap = 0

    total_time_B_to_parking_heatmap = 0

    total_time_B_to_parking_not_heatmap = 0

    total_distance_B_to_parking_not_heatmap = 0

    total_distance_B_to_parking_heatmap = 0

    # for edge in net.getEdges():

    #   print(edge.getID())

    for v, in_B in parked_in_B.items():

        if vehicle_use_heatmap[v] == 'True':

            num_use_heatmap += 1

        else:

            num_not_use_heatmap += 1

        if in_B == True:

            if vehicle_use_heatmap[v] == 'True':

                total_time_vehicle_heatmap += time_B_arrive[v] - time_A_start[v]

                num_parking_b_heatmap += 1

            else:

                total_time_vehicle_not_heatmap += time_B_arrive[v] - time_A_start[v]

                num_parking_b_not_heatmap += 1

        mid_point_B = get_midpoint(net.getEdge(origin_destinations[v]))

        mid_point_parking = get_midpoint(net.getEdge(edge_parked[v]))

        pedestrian_distance = calculate_distance(mid_point_B,

                                                 mid_point_parking)  # calculate the distance parking - point B as

        # the Euclidean distance of the midpoints of the two lanes

        if vehicle_use_heatmap[v] == 'True':

            total_time_B_to_parking_heatmap += time_parked[v] - time_B_arrive[v]

            total_distance_B_to_parking_heatmap += pedestrian_distance


        else:

            total_time_B_to_parking_not_heatmap += time_parked[v] - time_B_arrive[v]

            total_distance_B_to_parking_not_heatmap += pedestrian_distance

        # print(f"vehicle {v}, pedestrian distance: {pedestrian_distance} heatmap?: {vehicle_use_heatmap[v]}, parking in B?: {parked_in_B[v]}, time for parking from B:{time_parked[v] - time_B_arrive[v]}")

        # print(f"midpoint B:{mid_point_B}, midpoint parking:{mid_point_parking}, parking: {edge_parked[v]}, destinations:{destinations[v]}, time arrive to B and parking: {time_B_arrive[v] - time_A_start[v]}")

    t_a_to_b_pb_hm = None

    t_b_to_p_hm = None

    d_b_to_p_hm = None

    t_a_to_b_pb = None

    t_b_to_p = None

    d_b_to_p = None

    print("RESULTS------------------------")

    if num_use_heatmap != 0:

        if num_parking_b_heatmap != 0:
            t_a_to_b_pb_hm = round(float(total_time_vehicle_heatmap) / float(num_parking_b_heatmap), 2)

        t_b_to_p_hm = round(float(total_time_B_to_parking_heatmap) / float(num_use_heatmap), 2)

        d_b_to_p_hm = round(float(total_distance_B_to_parking_heatmap) / float(num_use_heatmap), 2)

        print(f"Average time to park in B for vehicles using heatmap: {t_a_to_b_pb_hm}s")

        print(f"Average parking search time for vehicles using heatmap: {t_b_to_p_hm}s")

        print(f"Average distance parking - point B using heatmap: {d_b_to_p_hm}m")

    if num_not_use_heatmap != 0:

        if num_parking_b_not_heatmap != 0:
            t_a_to_b_pb = round(float(total_time_vehicle_not_heatmap) / float(num_parking_b_not_heatmap), 2)

        t_b_to_p = round(float(total_time_B_to_parking_not_heatmap) / float(num_not_use_heatmap), 2)

        d_b_to_p = round(float(total_distance_B_to_parking_not_heatmap) / float(num_not_use_heatmap), 2)

        print(f"Average time to park in B for vehicles not using heatmap: {t_a_to_b_pb}s")

        print(f"Average parking search time for vehicles not using heatmap: {t_b_to_p}s")

        print(f"Average distance parking - point B not using heatmap: {d_b_to_p}m")

    print("END RESULTS---------------------")

    return t_a_to_b_pb_hm, t_b_to_p_hm, d_b_to_p_hm, t_a_to_b_pb, t_b_to_p, d_b_to_p


def generate_results(perc_hm, numVehicles, granularity, alfa, t_A_Bpb_hm, t_B_p_hm, d_B_p_hm, t_A_Bpb, t_B_p, d_B_p):
    # Define the column headers

    headers = [

        'usage_percentage_heatmap',

        'vehicles',

        'granularity',

        'alpha',

        'avg_parking_time_B_heatmap',

        'avg_search_parking_time_heatmap',

        'distance_parking_point_B_heatmap',

        'avg_parking_time_B',

        'avg_search_parking_time',

        'distance_parking_point_B'

    ]

    filename = 'results_data.csv'

    file_exists = os.path.isfile(filename)

    # Write data to the CSV file in append mode ('a')

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write headers only if the file does not exist

        if not file_exists:
            writer.writerow(headers)

        # Generate and write the data rows

        writer.writerow(
            [perc_hm, numVehicles, granularity, alfa, t_A_Bpb_hm, t_B_p_hm, d_B_p_hm, t_A_Bpb, t_B_p, d_B_p])

    print(f"File '{filename}' created successfully!")


def save_agent_csv(perc_use_heatmap, test_number_percentage, vehicle_id, A, B, p, heatmap, travel_time,

                   parking_search_time, pedestrian_distance, alfa, vehicle_number):
    # CSV file name

    file_name = 'data_agent.csv'

    # CSV headers (you can change these names as needed)

    header = ['perc_use_heatmap', 'test_number_percentage', 'vehicle_id', 'A', 'B', 'p', 'heatmap',

              'travel_time', 'parking_search_time', 'pedestrian_distance', 'alpha', 'vehicle_number']

    # Data to save

    data = [perc_use_heatmap, test_number_percentage, vehicle_id, A, B, p, heatmap,

            travel_time, parking_search_time, pedestrian_distance, alfa, vehicle_number]

    # Check if the file already exists or not

    try:

        with open(file_name, mode='a', newline='') as file:

            writer = csv.writer(file)

            # If the file is empty, write the header first

            file.seek(0, 2)  # Go to the end of the file

            if file.tell() == 0:  # If the file is empty

                writer.writerow(header)

            # Write the data

            writer.writerow(data)


    except Exception as e:

        print(f"Error while writing to CSV file: {e}")


# Function that starts a single SUMO simulation

"""def run_simulation(p, i):

    # Start SUMO with the --start option to start the simulation immediately

    traci.start(

        [sumoBinary, "-c", "parking_on_off_road.sumocfg", "--tripinfo-output", f"tripinfo_{p}_{i}.xml", "--start"])


    # Call the run function that handles the simulation logic

    run(p, i)


    # Close the connection with SUMO at the end of the simulation

    traci.close()



# Function that starts all simulations in parallel

def start_simulations_in_parallel(percentuali_heatmap, numero_test_percentuale):

    # Create a ProcessPoolExecutor to manage parallel processes

    with ProcessPoolExecutor() as executor:

        # List of futures to manage parallel executions

        futures = []


        # Loop to start simulations

        for p in percentuali_heatmap:

            for i in range(numero_test_percentuale):

                # Submit sends the execution of the run_simulation function to the process pool

                futures.append(executor.submit(run_simulation, p, i))


        # Wait for all simulations to finish

        for future in futures:

            future.result()  # This method blocks execution until the process is finished


"""


def populate_heatmap_parking_percentage(file_xml, percentage):  # new

    # Load and parse the XML file

    tree = ET.parse(file_xml)

    root = tree.getroot()

    parking_ids = []

    # Iterate over each 'parkingArea' element in the XML file

    for parking in root.findall('parkingArea'):
        id = parking.get('id')

        parking_ids.append(id)

    # Calculate the number of ids to select based on the percentage

    num_ids_selected = max(0, int((percentage / 100) * len(parking_ids)))

    random_ids = []

    if num_ids_selected > 0:
        # Randomly select the ids based on the calculated number

        random_ids = random.sample(parking_ids, num_ids_selected)



    return random_ids

def set_initial_vehicles_heatmap(vehicles, use_heatmap, id_parkings, heatmap):
    start_vehicles = []
    vehicles_in_parking = {}

    for v in vehicles:
        if use_heatmap[v] == 'True' and len(start_vehicles) < len(id_parkings):
            start_vehicles.append(v)

    for v, id_p in zip(start_vehicles, id_parkings):
        vehicles_in_parking[v] = id_p
        heatmap.update(True, id_p)
        heatmap.update(False, id_p)

    return vehicles_in_parking


def run(percentuali_heatmap, numero_test_percentuale, alfa, vehicle_number):
    parkage_map = HeatMap(xml_file='heat_map.xml', additional_file='parking_on_off_road.add.xml')
    parkage_map.update(True, real_parkages=True)
    # parkage_map.print_heatmap_values()
    # save real parking map
    parkage_map.save_heatmap_to_image('real_parkages.jpg', 'real parking areas', True)

    # list of all intermediate heatmap transitions (can create a GIF)
    historic_heatmap = []

    set_vehicle_point_A_B(vehicle_number)
    parking_list = traci.parkingarea.getIDList()  # list of parking areas

    parked_vehicles = {}  # parked vehicles with their respective parking areas
    recent_parking_vehicles = {}  # vehicles that recently left a parking area and their exit time
    parking_to_edge = {}  # parking areas and associated edges

    # Populate the dictionary
    for parking_id in parking_list:
        # Get the ID of the lane associated with the parking area
        lane_id = traci.parkingarea.getLaneID(parking_id)
        # Extract the edge ID from the lane
        edge_id = lane_id.split('_')[0]
        # Add to the dictionary
        parking_to_edge[parking_id] = edge_id

    # Path to the parking area configuration file
    parking_file = "parking_on_off_road.add.xml"
    # Dictionary to map parking areas to their capacities
    parking_capacity = {}
    # Parse the XML file
    tree = ET.parse(parking_file)
    root = tree.getroot()
    # Find all parking areas and get their capacities
    for parking_area in root.findall('parkingArea'):
        parking_id = parking_area.get('id')
        capacity = int(parking_area.get('roadsideCapacity'))
        parking_capacity[parking_id] = capacity
    # Print the dictionary

    # number of occupied spots per parking area, also considering vehicles about to park
    parking_car_parked = {}
    for parking in parking_list:
        parking_car_parked[parking] = 0

    COOLDOWN_PERIOD = 10000  # time difference between one parking and another (effectively no longer parks)
    exitLane = 'E86'

    # for each vehicle, keep track of how many times it has traversed an edge
    car_history_edge = defaultdict(dict)

    recent_changed_route_cars = {}
    REROUTE_PERIOD = 10000

    # exit_lane_list = []
    car_arrived_in_b = []  # list of vehicles that arrived at their point B

    """,preference """
    starting_lanes, destinations, use_heatmap = get_vehicle_point_A_B(
        percentuali_heatmap, vehicle_number)  # points B for each vehicle (parameter)
    vehicle_use_heatmap = copy.deepcopy(use_heatmap)  # used to retrieve analysis data
    origin_destinations = copy.deepcopy(destinations)  # copy the original points B

    print("Vehicles using heatmap: ", end='')
    cont_use_heatmap = 0
    for v, use in use_heatmap.items():
        if use == 'True':
            print(f"{v},", end='')
            cont_use_heatmap += 1

    perc_use_heatmap = round(float(cont_use_heatmap) / float(len(use_heatmap)), 2)  # actual percentage

    # print(destinations)
    net = sumolib.net.readNet("parking_on_off_road.net.xml")

    print("START HEATMAP")

    heatmap = HeatMap(xml_file='heat_map.xml', additional_file='parking_on_off_road.add.xml')
    print("END HEATMAP")

    delay_start = 10  # delay between one vehicle's departure and the next (parameter)
    current_delay_time = 0  # current delay
    time_A_start = {}  # simulation time when the vehicle departs (from point A)

    time_B_arrive = {}  # time when the vehicle arrives at B (only cases where it parked in B)
    parked_in_B = {}  # vehicle, boolean (True if the vehicle parked at its corresponding point B)
    time_parked = {}  # times when vehicles parked (vehicle, time)
    edge_parked = {}  # edge where the vehicle parked

    vehicle_index = 0  # index of the vehicle about to depart

    vehicle = list(starting_lanes.keys())[vehicle_index]
    st_lane = starting_lanes[vehicle]  # Get the departure lane

    # new
    vehicles_in_parking = set_initial_vehicles_heatmap(list(starting_lanes.keys()), use_heatmap,
                                                       populate_heatmap_parking_percentage(
                                                           'parking_on_off_road.add.xml', 20), heatmap)
    print(vehicles_in_parking)

    STOP_PARKAGE_INIT = 20

    if len(vehicles_in_parking) == 0:
        print(f"Starting vehicle: {vehicle}")
        # Create a route for the vehicle based on the edge (st_lane)
        traci.route.add(routeID=f"route_{vehicle}", edges=[st_lane])
        traci.vehicle.add(
            vehID=vehicle,
            routeID=f"route_{vehicle}",  # Route created dynamically for each vehicle
            departPos="0",  # Initial position on the lane
            departSpeed="0.1"  # Maximum speed at departure
        )
        traci.vehicle.setMaxSpeed(vehicle, 5.0)
        vehicle_index += 1
        time_A_start[vehicle] = traci.simulation.getTime()

    else:
        for vehicle in vehicles_in_parking:
            parking_id = vehicles_in_parking[vehicle]
            start_lane = traci.parkingarea.getLaneID(parking_id).split('_')[0]
            start_position = traci.parkingarea.getStartPos(parking_id)

            traci.route.add(routeID=f"route_{vehicle}", edges=[start_lane])
            # Add the vehicle to the simulation and park it
            traci.vehicle.add(
                vehID=vehicle,
                routeID=f"route_{vehicle}",
                depart="now",
                departPos=start_position,  # Initial position on the lane
                departSpeed="0.1"
            )
            traci.vehicle.setMaxSpeed(vehicle, 5.0)
            # Park the vehicle in the specified parking area
            traci.vehicle.setParkingAreaStop(vehicle, parking_id, STOP_PARKAGE_INIT)
            time_A_start[vehicle] = traci.simulation.getTime() + STOP_PARKAGE_INIT
            # starting_lanes.pop(vehicle, None)

        """else:
            # Create a route for the vehicle based on the edge (st_lane)
            traci.route.add(routeID=f"route_{vehicle}", edges=[st_lane])
            traci.vehicle.add(
                vehID=vehicle,
                routeID=f"route_{vehicle}",  # Route created dynamically for each vehicle
                departPos="0",  # Initial position on the lane
                departSpeed="0.1"  # Maximum speed at departure
            )
        vehicle_index += 1"""

    # data structures to retrieve analysis data

    points_A = []  # points A of the vehicles
    points_parking = []  # parking points of the vehicles
    # vehicle_use_heatmap to check heatmap usage

    while traci.simulation.getMinExpectedNumber() > 0:
        # print(f"current number of vehicles: {traci.vehicle.getIDList()}")
        # if len(traci.vehicle.getIDList()) == 0:
        # break

        # print(f"expected vehicles:{ traci.simulation.getMinExpectedNumber()}")

        current_time = traci.simulation.getTime()
        # print(f"Current time: {current_time} seconds")

        # to fix
        # print(f"vehicle_index: {vehicle_index}")
        if current_delay_time % delay_start == 0 and vehicle_index < len(starting_lanes):
            vehicle = list(starting_lanes.keys())[vehicle_index]  # Get the vehicle
            # print(list(starting_lanes.keys()))
            # print(vehicles_in_parking)
            print(f"considering vehicle {vehicle}")
            while vehicle in vehicles_in_parking:
                vehicle_index += 1
                vehicle = list(starting_lanes.keys())[vehicle_index]  # Get the next vehicle

            # print(f"Starting vehicle: {vehicle}")

            # new

            # parking_id = vehicles_in_parking[vehicle]
            start_lane = starting_lanes[vehicle]
            # start_position = traci.parkingarea.getStartPos(parking_id)

            traci.route.add(routeID=f"route_{vehicle}", edges=[start_lane])

            traci.vehicle.add(
                vehID=vehicle,
                routeID=f"route_{vehicle}",
                depart="now",
                departPos=0,  # Initial position on the lane
                departSpeed="0.1"
            )
            time_A_start[vehicle] = current_time

            traci.vehicle.setMaxSpeed(vehicle, 5.0)

            vehicle_index += 1  # Increment only after adding the vehicle



        current_delay_time += 1

        traci.simulationStep()

        if len(traci.vehicle.getIDList()) == 0:
            print("All vehicles have exited: simulation ended!")
            break

        for vehicle_id in traci.vehicle.getIDList():
            if use_heatmap[vehicle_id] == 'True':
                print(f"Vehicle {vehicle_id} uses heatmap")
                """,preference"""
                heatmap.direct_vehicle_to_best_parking(vehicle_id, destinations, parkage_map, net, alfa)
                use_heatmap[vehicle_id] = None

            if vehicle_id not in car_arrived_in_b:
                # print(f"arrival: {traci.vehicle.getRoute(vehicle_id)[-1]} newdest: {destinations[vehicle_id]}")
                # if it has just arrived
                if traci.vehicle.getLaneID(vehicle_id).split('_')[0] == destinations[vehicle_id]:
                    car_arrived_in_b.append(vehicle_id)
                    print(f"Vehicle {vehicle_id} arrived at B: {destinations[vehicle_id]}")
                    time_B_arrive[vehicle_id] = current_time

                    # print(f"Vehicle {vehicle_id} arrived at destination B" )
                elif traci.vehicle.getRoute(vehicle_id)[-1] != destinations[vehicle_id]:
                    traci.vehicle.changeTarget(vehicle_id, destinations[vehicle_id])
                    for edge in traci.vehicle.getRoute(vehicle_id):
                        if edge not in car_history_edge[vehicle_id]:
                            car_history_edge[vehicle_id][edge] = 0
                        car_history_edge[vehicle_id][edge] += 1

                if not use_heatmap[vehicle_id] == None:
                    traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))  # do not use heatmap
                else:
                    traci.vehicle.setColor(vehicle_id, (255, 0, 255, 255))  # use heatmap

            if vehicle_id in car_arrived_in_b:
                if not is_vehicle_parked(vehicle_id):

                    # list of vehicles that are exiting
                    # if traci.vehicle.getRoadID(vehicle_id) == exitLane and vehicle_id not in exit_lane_list:
                    #    exit_lane_list.append(vehicle_id)

                    if recent_changed_route_cars.get(vehicle_id) != None and current_time - recent_changed_route_cars[
                        vehicle_id] > REROUTE_PERIOD:
                        del recent_changed_route_cars[vehicle_id]
                    # if the vehicle has just turned, there is no more control on double resetting
                    if recent_changed_route_cars.get(vehicle_id) != None:
                        # print(f"vehicle edge: {vehicle_id} current edge: {traci.vehicle.getRoadID(vehicle_id)} destination: {traci.vehicle.getRoute(vehicle_id)[-1]}")
                        # if at the destination
                        if traci.vehicle.getRoadID(vehicle_id) == traci.vehicle.getRoute(vehicle_id)[-1]:
                            del recent_changed_route_cars[vehicle_id]
                        # print(f"Vehicle {vehicle_id} can change route again")
                        # print(f"occurrences: { car_history_edge[vehicle_id]}")

                    if vehicle_id in parked_vehicles:
                        if is_exit_Parkage(vehicle_id, parked_vehicles[vehicle_id], parking_to_edge):
                            # print(f"Vehicle {vehicle_id} exited parking area {parked_vehicles[vehicle_id]}")
                            parking_car_parked[
                                parked_vehicles[vehicle_id]] -= 1  # update the number of parked vehicles

                            if use_heatmap[
                                vehicle_id] == None:  # if it uses it (I set to None those that have used it)
                                heatmap.update(False, parked_vehicles[vehicle_id])

                            del parked_vehicles[vehicle_id]  # remove from parked vehicles
                            recent_parking_vehicles[vehicle_id] = current_time  # add to recently exited vehicles

                            mid_point_B = get_midpoint(net.getEdge(origin_destinations[vehicle_id]))
                            mid_point_parking = get_midpoint(net.getEdge(edge_parked[vehicle_id]))
                            pedestrian_distance = calculate_distance(mid_point_B,
                                                                mid_point_parking)  # calculate the distance parking - point B as

                            save_agent_csv(perc_use_heatmap, numero_test_percentuale, vehicle_id,
                                           starting_lanes[vehicle_id]
                                           , destinations[vehicle_id], edge_parked[vehicle_id],
                                           vehicle_use_heatmap[vehicle_id],
                                           time_B_arrive[vehicle_id] - time_A_start[vehicle_id],
                                           time_parked[vehicle_id] -
                                           time_B_arrive[vehicle_id], round(pedestrian_distance, 2), alfa, vehicle_number)

                            # delete last reroute if it exists
                            # if the destination is not the lane where the vehicle is now
                            # print(
                            # f"Vehicle {vehicle_id} road {traci.vehicle.getLaneID(vehicle_id).split('_')[0]} destination {traci.vehicle.getRoute(vehicle_id)[-1]}")

                            if traci.vehicle.getRoute(vehicle_id)[-1] != traci.vehicle.getLaneID(vehicle_id).split('_')[
                                0]:
                                # if traci.vehicle.getRoute(vehicle_id)[-1] != exitLane:
                                # print(
                                #   f"Destination {traci.vehicle.getRoute(vehicle_id)[-1]} for vehicle {vehicle_id} is to be deleted")
                                car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]] -= 1
                                if car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]] == 0:
                                    del car_history_edge[vehicle_id][traci.vehicle.getRoute(vehicle_id)[-1]]

                            # once the vehicle has exited the parking area, direct it to an exit road

                            """from_edge_obj = net.getEdge(traci.vehicle.getLaneID(vehicle_id).split('_')[0])
                            to_edge_obj = net.getEdge(exitLane)
                            route = net.getShortestPath(from_edge_obj, to_edge_obj)
                            path = route[0]
                            edge_ids = [edge.getID() for edge in path]

                            traci.vehicle.setRoute(vehicle_id, edge_ids)"""
                            traci.vehicle.changeTarget(vehicle_id, exitLane)
                            traci.vehicle.setColor(vehicle_id, (0, 255, 0, 255))

                    if vehicle_id not in recent_parking_vehicles:  # if the vehicle has not recently left a parking area
                        # traci.vehicle.setColor(vehicle_id, (255, 0, 0, 255))
                        for parking_id in parking_list:
                            # if the vehicle is close enough to the parking area
                            if is_near_parkage(vehicle_id, parking_id,
                                               parking_to_edge) and vehicle_id not in parked_vehicles:
                                if park_vehicle(vehicle_id, parking_id, parking_car_parked, parking_capacity,
                                                parked_vehicles):
                                    print(f"Vehicle {vehicle_id} parked")
                                    time_parked[vehicle_id] = current_time
                                    edge_parked[vehicle_id] = traci.parkingarea.getLaneID(parking_id).split('_')[0]
                                    if parking_to_edge[parking_id] == destinations[vehicle_id]:
                                        print(f"Vehicle {vehicle_id} parked at point B")
                                        parked_in_B[vehicle_id] = True
                                    else:
                                        print(
                                            f"Vehicle {vehicle_id} parked at a point different from B (origin: {destinations[vehicle_id]}), parking {parking_to_edge[parking_id]}")
                                        parked_in_B[vehicle_id] = False

                                    if use_heatmap[vehicle_id] == None:
                                        heatmap.update(True, parked_vehicles[vehicle_id])
                                else:
                                    # print("Vehicle not parked: no more space! Looking for a new parking area")

                                    # find the nearest free parking area
                                    """nearestRoute = find_nearest_parkage(parking_id, parking_list, parking_to_edge, parking_capacity)
                                    print(nearestRoute)
                                    if nearestRoute != None:
                                        path = nearestRoute[0]
                                        edge_ids = [edge.getID() for edge in path]
                                        print(edge_ids)
                                        traci.vehicle.setRoute(vehicle_id, edge_ids)"""

                        # random rerouting logic
                        destination_edge = traci.vehicle.getRoute(vehicle_id)[-1]
                        if destination_edge != exitLane:
                            # check if the vehicle is near a junction
                            near_junction, junctionID = is_vehicle_near_junction(vehicle_id, net, 25)
                            if near_junction:
                                all_edges = traci.edge.getIDList()
                                # Filter to get only real edges
                                real_edges = [edge for edge in all_edges if not edge.startswith(':')]
                                # print(f"{traci.vehicle.getLaneID(vehicle_id).split('_')[0]} {real_edges}")

                                # if the vehicle is not on the junction
                                if traci.vehicle.getLaneID(vehicle_id).split('_')[0] in real_edges:

                                    if recent_changed_route_cars.get(vehicle_id) == None:
                                        # print(f"Vehicle {vehicle_id} is near junction {junctionID}")
                                        # calculate reachable edges
                                        laneID = traci.vehicle.getLaneID(vehicle_id)
                                        reachable_edges = get_reachable_edges_from_lane(laneID)
                                        # print(f"Vehicle {vehicle_id} on lane {laneID} can reach edges: {reachable_edges}")
                                        if reachable_edges:
                                            selected_lane, recent_changed_route_cars[
                                                vehicle_id], pass_count = set_vehicle_route(vehicle_id,
                                                                                            car_history_edge,
                                                                                            reachable_edges)
                                            # print(f"number of passes: {pass_count}")
                                            # print(f"Vehicle {vehicle_id} routed to lane {selected_lane}")





                    else:

                        if current_time - recent_parking_vehicles[vehicle_id] > COOLDOWN_PERIOD:
                            del recent_parking_vehicles[vehicle_id]

        deep_copy_heatmap = copy.deepcopy(heatmap)

        historic_heatmap.append(deep_copy_heatmap)

    # calculate various data

    print("Starting data analysis collection...")

    print(f"Total vehicles: {len(use_heatmap)}, of which {cont_use_heatmap} use the heatmap")

    print(f"Percentage {float(cont_use_heatmap) / float(len(use_heatmap)) * 100:.2f}%")

    # print(parked_in_B)

    time_A_B_pB_hm, time_B_p_hm, distance_B_p_hm, time_A_B_pB, time_B_p, distance_B_p = calculate_data_analysis(

        time_A_start, time_B_arrive, parked_in_B, vehicle_use_heatmap, time_parked, edge_parked, origin_destinations,

        net)

    generate_results(f"{float(cont_use_heatmap) / float(len(use_heatmap)) * 100:.2f}%", len(use_heatmap),

                     heatmap._read_area_size_from_xml(xml_file='heat_map.xml'), alfa,

                     time_A_B_pB_hm,

                     time_B_p_hm, distance_B_p_hm, time_A_B_pB, time_B_p, distance_B_p)

    print("End of data analysis collection")

    # Create GIF of the heatmap-----------------

    # Temporary folder for images

    temp_dir = "images_GIF"

    os.makedirs(temp_dir, exist_ok=True)

    image_filenames = []

    N = 70  # Save an image every 10 steps

    """print(f"Number of heatmap frames {len(historic_heatmap)}")


    # Save each heatmap as an image

    for i, heatmap in enumerate(historic_heatmap):

        if i % N == 0:  # Only every N steps

            filename = f"{temp_dir}/heatmap_{i}.png"

            historic_heatmap[i].save_heatmap_to_image(filename)

            image_filenames.append(filename)

            print(f"saved image {i}")


    filename = f"{temp_dir}/heatmap_{(len(historic_heatmap)/N)*N+N}.png"

    historic_heatmap[len(historic_heatmap)-1].save_heatmap_to_image(filename)

    image_filenames.append(filename)

    print(f"saved image {(len(historic_heatmap)/N)*N+N}")


    print("end saving GIF images")

    # Create a GIF from the images

    with imageio.get_writer('heatmap_animation.gif', mode='I', duration=0.5) as writer:

        for filename in image_filenames:    

            image = imageio.imread(filename)

            writer.append_data(image)


    # Remove temporary images

    for filename in image_filenames:

        os.remove(filename)


    # Delete the temporary folder

    shutil.rmtree(temp_dir)

    print("GIF created successfully!")


    #end heatmap GIF creation-------------------------"""

    traci.close()

    # save heatmap

    heatmap.save_heatmap_to_image('heatmap.jpg')

    print(f"Original destinations:{origin_destinations}")

    print(f"New destinations:{destinations}")

    # print(f"Number of exited vehicles: {len(exit_lane_list)}")

    # print(sorted(exit_lane_list, key=extract_number))

    # print("Vehicle pass history")

    # for v in car_history_edge:

    #    print(f"vehicle {v} : edges {car_history_edge[v]}")


def simulate_percentage(p, i):
    # Start SUMO with the --start option to start the simulation immediately

    traci.start(

        [sumoBinary, "-c", "parking_on_off_road.sumocfg", "--tripinfo-output", f"tripinfo_{p}_{i}.xml",

         "--start", "--quit-on-end"])

    # Call the run function that handles the simulation logic

    run(p, i)


# main entry point

if __name__ == '__main__':

    options = get_options()

    # check binary

    if options.nogui:

        sumoBinary = checkBinary('sumo')

    else:

        sumoBinary = checkBinary('sumo-gui')

    # 0,0.25,0.5,0.75,1

    percentuali_heatmap = [1]  # then also put 0

    numero_test_percentuale = 10  # 10

    # 0.2,0.3,0.4,0.5,0.6,0.7,0.8

    alfa = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
            0.8]  # coefficient of weights to calculate the score (the smaller it is, the more weight is given to distance)

    # start_simulations_in_parallel(percentuali_heatmap, numero_test_percentuale)

    vehicle_number = [300]  # ,250,300

    for p in percentuali_heatmap:

        for a in alfa:

            for v_num in vehicle_number:

                for i in range(numero_test_percentuale):
                    # Start SUMO with the --start option to start the simulation immediately

                    traci.start(

                        [sumoBinary, "-c", "parking_on_off_road.sumocfg", "--tripinfo-output", f"tripinfo_{p}_{i}.xml",

                         "--start", "--quit-on-end"])

                    # Call the run function that handles the simulation logic

                    run(p, i, a, v_num)

    # Number of parallel processes, you can modify it depending on the number of cores of your machine

    """numero_processi = 3


    # Parallelize the simulations

    with concurrent.futures.ProcessPoolExecutor(max_workers=numero_processi) as executor:

        futures = []

        for p in percentuali_heatmap:

            for i in range(numero_test_percentuale):

                futures.append(executor.submit(simulate_percentage, p, i))


        # Wait for all simulations to complete

        concurrent.futures.wait(futures)"""