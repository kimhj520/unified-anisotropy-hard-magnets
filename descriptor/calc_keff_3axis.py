# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:07:47 2024

@author: hjkim
"""

import os
import csv
import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

def calculate_volume(file_path):
    vectors = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[2:5]  # read 3,4,5th lines from poscar
        for line in lines:
            vectors.append([float(x) for x in line.split()])

    # convert vector to numpy array
    vec1, vec2, vec3 = np.array(vectors)
    
    # calculate volume by 3 vectors
    volume = abs(np.dot(vec1, np.cross(vec2, vec3)))
    return volume

def update_energies_csv(input_csv_path, output_csv_path):
    rows = []
    
    with open(input_csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            # read mpid and energies
            main_id = row[0]
            #f_values = [float(x) for x in row[2:] if '(' not in x and ')' not in x]
            f_values = [float(x) for x in row[2:] if x.strip() and '(' not in x and ')' not in x]
            poscar_path = os.path.join(main_id, 'a', 'POSCAR')
            
            # sort out the lowest and second lowest energy value and calculate mca energy (tolerance=1e-5)
            sorted_f_values = sorted(f_values)
            if len(f_values) == 2:
                # No sorting: subtract in original order
                diff = (f_values[0] - f_values[1]) * 1000000
            elif len(f_values) == 3:
                sorted_f_values = sorted(f_values)
                diff = (sorted_f_values[1] - sorted_f_values[0]) * 1000000

            # unit conversion to MJ/m^3
            if os.path.exists(poscar_path):
                volume = calculate_volume(poscar_path)
                new_value = 0.16 * diff / volume
            else:
                new_value = None  
            
            while len(row) < 7:
                row.append('')
    
            # append calculated mca energy
            row[5] = diff
            row[6] = new_value
            rows.append(row)
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in rows:
            csv_writer.writerow(row)
    
    print(f"CSV file '{output_csv_path}' is written.")

def calculate_mag_in_T(mpid):
    doc = mpr.materials.magnetism.search(material_ids=mpid, fields = ['total_magnetization', 'total_magnetization_normalized_formula_units', 'total_magnetization_normalized_vol', 'volume'])
    vol = doc[0].volume
    mag_per_uc = doc[0].total_magnetization_normalized_formula_units
    mag_in_T = (mag_per_uc * 9.274 * 1.257)/vol

    return mag_in_T
    
def update_magmom_csv(output_csv_path, additional_csv_path):
    with open(output_csv_file, 'r', newline='') as infile, open(additional_csv_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            # 각 행의 첫 번째 원소에 custom_function 적용
            mpid = row[0]
            mag_in_T = calculate_mag_in_T(mpid)
            
            # 결과를 마지막 열에 추가
            row.append(mag_in_T)
            
            # 수정된 행을 새로운 CSV 파일에 작성
            writer.writerow(row)
            
    print(f"CSV file '{additional_csv_path}' is written.")
        
if __name__ == "__main__":
    mpr = MPRester(api_key="YOUR_MP_API_KEY")
    # read mca.csv file to update it
    input_csv_path = 'energy_values.csv'
    output_csv_path = 'update_calculated_mca.csv'
    additional_csv_path = 'update_calculated_mca_mag.csv'
    
    update_energies_csv(input_csv_path, output_csv_path)