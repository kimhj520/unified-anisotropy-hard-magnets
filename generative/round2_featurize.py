# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:57:41 2025

@author: hjkim
"""
#%%

from pymatgen.core import Structure
from mendeleev import element
import numpy as np
import re
import pandas as pd
import os
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

def get_last_electron_counts(electron_config):
    # 각 오비탈에 해당하는 마지막 전자수를 저장할 딕셔너리 초기화
    electron_counts = {'s': 0, 'p': 0, 'd': 0, 'f': 0}
    
    # 오비탈에 최대 전자수를 설정
    max_electrons = {'s': 2, 'p': 6, 'd': 10, 'f': 14}
    
    # 정규 표현식을 사용해 오비탈과 해당 전자수 찾기
    orbitals = re.findall(r'(\d)([spdf])(\d+)', electron_config)
    
    # 마지막으로 등장하는 각 오비탈의 전자수만 업데이트
    for _, orbital_type, electron_count in orbitals:
        count = int(electron_count)
        # 해당 오비탈이 꽉 찬 경우 0을 설정, 그렇지 않으면 실제 전자수
        #electron_counts[orbital_type] = 0 if count == max_electrons[orbital_type] else count
        electron_counts[orbital_type] = count
    
    return electron_counts


def count_unfilled_electrons(electron_config):
    # 부껍질의 최대 전자 수 정의
    max_electrons = {'s': 2, 'p': 6, 'd': 10, 'f': 14}
    # 각 부껍질의 채워지지 않은 전자 수를 저장할 딕셔너리 초기화
    unfilled_electrons = {'s': 0, 'p': 0, 'd': 0, 'f': 0}
    
    # 정규 표현식을 사용하여 오비탈과 전자 수 추출
    orbitals = re.findall(r'(\d)([spdf])(\d+)', electron_config)
    
    for _, orbital_type, electron_count in orbitals:
        electron_count = int(electron_count)
        # 최대 전자 수에서 현재 전자 수를 뺀 값을 구해 채워지지 않은 전자 수 저장
        unfilled_electrons[orbital_type] = max_electrons[orbital_type] - electron_count
    
    return unfilled_electrons

def update_dict_keys(input_dict):
    updated_dict = {}
    
    for key, value in input_dict.items():
        if isinstance(value, dict):
            # dict인 경우, 그 dict의 values에서 None을 제외한 값들의 중간값을 구함
            valid_values = [v for v in value.values() if v is not None]
            if valid_values:
                mid_value = np.median(valid_values)
                updated_dict[key] = mid_value  # 중간값을 key의 value로 설정
            else:
                updated_dict[key] = None  # 모든 값이 None일 경우 None
        else:
            # dict가 아닌 경우 그대로 추가
            updated_dict[key] = value
            
    return updated_dict

atomic_num = {}
mendeleev_num = {}
column = {}
row = {}
volume = {}
spin_only_magmom = {}
mass = {}
EN = {}
r = {}
#E_cohesive = {}
E_i = {}
EA = {}
polarizability = {}
density = {}
Tb = {}
Tm = {}
s_electron = {}
p_electron = {}
d_electron = {}
f_electron = {}
unfilled_s_electron = {}
unfilled_p_electron = {}
unfilled_d_electron = {}
unfilled_f_electron = {}
unpaired_elec = {}

for i in range(1, 95):
    elem = element(i)
    
    atomic_num[elem.symbol] = elem.atomic_number
    mendeleev_num[elem.symbol] = elem.mendeleev_number
    column[elem.symbol] = elem.group_id
    row[elem.symbol] = elem.period
    volume[elem.symbol] = elem.atomic_volume
    spin_only_magmom[elem.symbol] = elem.ec.spin_only_magnetic_moment()
    mass[elem.symbol] = elem.mass
    EN[elem.symbol] = elem.electronegativity('mulliken')
    
    r[elem.symbol] = elem.atomic_radius
    E_i[elem.symbol] = elem.ionenergies[1]
    EA[elem.symbol] = elem.electron_affinity
    polarizability[elem.symbol] = elem.dipole_polarizability
    density[elem.symbol] = elem.density
    Tb[elem.symbol] = elem.boiling_point
    Tb = update_dict_keys(Tb)
    Tm[elem.symbol] = elem.melting_point
    Tm = update_dict_keys(Tm)
    
    e_config = get_last_electron_counts(str(elem.ec))
    unfilled_e_config = count_unfilled_electrons(str(elem.ec))
    
    s_electron[elem.symbol] = e_config['s']
    p_electron[elem.symbol] = e_config['p']
    d_electron[elem.symbol] = e_config['d']
    f_electron[elem.symbol] = e_config['f']
    
    unfilled_s_electron[elem.symbol] = unfilled_e_config['s']
    unfilled_p_electron[elem.symbol] = unfilled_e_config['p']
    unfilled_d_electron[elem.symbol] = unfilled_e_config['d']
    unfilled_f_electron[elem.symbol] = unfilled_e_config['f']
    
    unpaired_elec[elem.symbol] = elem.ec.unpaired_electrons()
    
#%% 각 구조별 cif파일 읽고 stoichiometric/compositional/structural feature 생성 (X)

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from dscribe.descriptors import SineMatrix
from pymatgen.core import Structure, Element, Composition

# dscribe System.from_atoms uses ASE methods removed in newer ASE (_get_constraints).
# SineMatrix only needs atomic numbers, positions, and cell -> patch with a safe builder.
from dscribe.core.system import System as _DSystem
def _safe_from_atoms(atoms):
    return _DSystem(numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions(),
                    cell=atoms.get_cell(), pbc=True)
_DSystem.from_atoms = staticmethod(_safe_from_atoms)

mpr = MPRester('YOUR_MP_API_KEY')

def calculate_ratios(a, b, c):
    # a, b, c 값을 작은 것부터 큰 것 순으로 배열
    sorted_values = sorted([a, b, c])

    # a, b, c 순서에 맞는 비율 계산
    ratio_ab = sorted_values[1] / sorted_values[0]  # b/a
    ratio_ac = sorted_values[2] / sorted_values[0]  # c/a
    ratio_bc = sorted_values[2] / sorted_values[1]  # c/b

    # 비율들의 산술 평균 계산
    arithmetic_mean = (ratio_ab + ratio_ac + ratio_bc) / 3

    # 비율들의 기하 평균 계산
    geometric_mean = (ratio_ab * ratio_ac * ratio_bc) ** (1 / 3)

    return arithmetic_mean, geometric_mean

def lattice_cv(a, b, c):
    lattice_constants = np.array([a, b, c])
    mean_val = lattice_constants.mean()
    std_val = lattice_constants.std()
    return std_val / mean_val if mean_val != 0 else 0

def anisotropy_factor(a, b, c):
    lattice = [a, b, c]
    mean_val = sum(lattice) / 3
    numerator = sum((x - mean_val)**2 for x in lattice)
    return (numerator / (3 * mean_val**2))**0.5 if mean_val != 0 else 0

def structure_matcher(cif_file):
    """
    주어진 CIF 파일의 구조가 Materials Project에 있는 해당 구조와 동일한지 확인하는 함수.
    """
    # 1. CIF 파일에서 구조 읽기
    structure = Structure.from_file(cif_file)

    # 2. 파일 이름에서 {자연수} 부분 추출 (예: '12345.cif' -> 'mp-12345')
    cif_id = os.path.basename(cif_file).replace(".cif", "")
    mp_id = f"mp-{cif_id}"  # Materials Project ID 변환

    # 3. Materials Project에서 해당 구조 검색
    docs = mpr.materials.search(material_ids=[mp_id])

    # 4. 검색된 구조가 존재하는지 확인
    if not docs:
        print(f"❌ {mp_id}: Materials Project에서 해당 구조를 찾을 수 없습니다.")
        return None  # None 반환 (비교 불가)

    # 5. 검색된 구조의 결정 구조(Structure) 불러오기
    mp_structure = docs[0].structure  # 검색된 첫 번째 결과의 구조

    # 6. CIF 파일의 구조와 Materials Project의 구조 비교
    matcher = StructureMatcher()  # 구조 비교 객체 생성
    is_match = matcher.fit(structure, mp_structure)

    '''
    if is_match:
        print(f"✅ {mp_id}: 구조가 일치합니다.")
    else:
        print(f"❌ {mp_id}: 구조가 다릅니다.")
    '''
    return is_match

def get_target_elements():
    transition_metals = [el for el in Element if el.is_transition_metal and el.Z <= 83]
    lanthanides = [el for el in Element if el.is_lanthanoid]
    actinides = [el for el in Element if el.is_actinoid and el.Z <= 94]  # Pu까지

    # 중복 제거 및 원자번호 정렬
    unique_elements = {el.symbol: el for el in transition_metals + lanthanides + actinides}
    sorted_elements = sorted(unique_elements.values(), key=lambda el: el.Z)
    return [el.symbol for el in sorted_elements]

avg = lambda lst: (
    sum(x for x in lst if x is not None) / 
    sum(1 for x in lst if x is not None)
    if sum(1 for x in lst if x is not None) > 0 
    else 0
)

max_val = lambda lst: (
    max(x for x in lst if x is not None) 
    if any(x is not None for x in lst) 
    else 0
)

min_val = lambda lst: (
    min(x for x in lst if x is not None) 
    if any(x is not None for x in lst) 
    else 0
)

max_diff = lambda lst: (
    max(x for x in lst if x is not None) - 
    min(x for x in lst if x is not None) 
    if len([x for x in lst if x is not None]) > 1 
    else 0
)

min_diff = lambda lst: (
    min(
        abs(x - y) 
        for i, x in enumerate(lst) 
        for y in lst[i+1:] 
        if x is not None and y is not None
    ) 
    if len([x for x in lst if x is not None]) > 1 
    else 0
)

std = lambda lst: (
    (sum(
        (x - (sum(x for x in lst if x is not None) / len([x for x in lst if x is not None]))) ** 2 
        for x in lst if x is not None
    ) / len([x for x in lst if x is not None])) ** 0.5 
    if len([x for x in lst if x is not None]) > 0 
    else 0
)

weighted_avg = lambda props, count: (
    sum(prop * cnt for prop, cnt in zip(props, count) if prop is not None) / 
    sum(cnt for prop, cnt in zip(props, count) if prop is not None)
    if len([prop for prop in props if prop is not None]) > 0 
    else 0
)

weighted_max = lambda props, count: (
    max(prop * cnt for prop, cnt in zip(props, count) if prop is not None) 
    if len([prop for prop in props if prop is not None]) > 0 
    else 0
)

weighted_min = lambda props, count: (
    min(prop * cnt for prop, cnt in zip(props, count) if prop is not None) 
    if len([prop for prop in props if prop is not None]) > 0 
    else 0
)

weighted_max_diff = lambda props, count: (
    max([prop * cnt for prop, cnt in zip(props, count) if prop is not None]) - 
    min([prop * cnt for prop, cnt in zip(props, count) if prop is not None]) 
    if len([prop for prop in props if prop is not None]) > 1 
    else 0
)
'''
weighted_min_diff = lambda props, count: (
    min(
        abs(prop1 * cnt1 - prop2 * cnt2) 
        for i, (prop1, cnt1) in enumerate(zip(props, count)) 
        for prop2, cnt2 in zip(props[i+1:], count[i+1:]) 
        if prop1 is not None and prop2 is not None
    ) 
    if len([prop for prop in props if prop is not None]) > 1 
    else 0
)
'''
def weighted_min_diff(props, count):
    diffs = [
        abs(prop1 * cnt1 - prop2 * cnt2)
        for i, (prop1, cnt1) in enumerate(zip(props, count))
        for prop2, cnt2 in zip(props[i+1:], count[i+1:])
        if prop1 is not None and prop2 is not None
    ]
    return min(diffs) if diffs else 0

weighted_std = lambda props, count: (
    (sum(
        ((prop * cnt) - (sum((prop * cnt) for prop, cnt in zip(props, count) if prop is not None) / 
        len([prop for prop in props if prop is not None]))) ** 2 
        for prop, cnt in zip(props, count) if prop is not None
    ) / len([prop for prop in props if prop is not None])) ** 0.5 
    if len([prop for prop in props if prop is not None]) > 0 
    else 0
)


def analyze_cif_properties(cif_file):
    # CIF 파일 읽기
    structure = Structure.from_file(cif_file)
    
    # 원소 빈도수 계산
    element_counts = structure.composition.get_el_amt_dict()
    total_atoms = sum(element_counts.values())
    
    # 특성 수집을 위한 초기화
    s_electrons, p_electrons, d_electrons, f_electrons = [], [], [], []
    unfilled_s, unfilled_p, unfilled_d, unfilled_f = [], [], [], []
    unpaired_electrons, magnetic_moments, atomic_radii, electronegativities = [], [], [], []
    atomic_number, mendeleev, masses, ionE, affin, dipole, den, boiling, melting = [], [], [], [], [], [], [], [], []
    group, period, vol = [], [], []
    
    # 각 원소에 대해 특성 수집
    for elem, count in element_counts.items():
        
        s_electrons.append(s_electron[elem])
        p_electrons.append(p_electron[elem])
        d_electrons.append(d_electron[elem])
        f_electrons.append(f_electron[elem])
        
        unfilled_s.append(unfilled_s_electron[elem])
        unfilled_p.append(unfilled_p_electron[elem])
        unfilled_d.append(unfilled_d_electron[elem])
        unfilled_f.append(unfilled_f_electron[elem])
        
        unpaired_electrons.append(unpaired_elec[elem])
        magnetic_moments.append(spin_only_magmom[elem])
        electronegativities.append(EN[elem])
        atomic_radii.append(r[elem])
        
        atomic_number.append(atomic_num[elem])
        mendeleev.append(mendeleev_num[elem])
        masses.append(mass[elem])
        ionE.append(E_i[elem])
        affin.append(EA[elem])
        dipole.append(polarizability[elem])
        den.append(density[elem])
        boiling.append(Tm[elem])
        melting.append(Tb[elem]) 
        
        group.append(column[elem])
        period.append(row[elem])
        vol.append(volume[elem])
    '''
    if structure_matcher(cif_file):
        cif_id = os.path.basename(cif_file).replace(".cif", "")
        mp_id = f"mp-{cif_id}"
        magmoms = mpr.materials.magnetism.search(material_ids=mp_id, fields='magmoms')[0].magmoms
    '''
    # 모든 특성에 대해 평균, 표준 편차, 최댓값 차 계산
    '''
    properties = [s_electrons, p_electrons, d_electrons, f_electrons, 
                  unfilled_s, unfilled_p, unfilled_d, unfilled_f,
                  unpaired_electrons, magnetic_moments, atomic_radii, electronegativities,
                  atomic_number, mendeleev, masses, ionE, affin, dipole, den, boiling, melting,
                  group, period, vol, magmoms]
    '''
    properties = [s_electrons, p_electrons, d_electrons, f_electrons, 
                  unfilled_s, unfilled_p, unfilled_d, unfilled_f,
                  unpaired_electrons, magnetic_moments, atomic_radii, electronegativities,
                  atomic_number, mendeleev, masses, ionE, affin, dipole, den, boiling, melting,
                  group, period, vol]
    
    results = []
    for prop in properties:
        prop_avg = avg(prop)
        prop_max = max_val(prop)
        prop_min = min_val(prop)
        prop_max_diff = max_diff(prop)
        prop_min_diff = min_diff(prop)
        prop_std = std(prop)
        
        prop_wavg = weighted_avg(prop, list(element_counts.values()))
        prop_wmax = weighted_max(prop, list(element_counts.values()))
        prop_wmin = weighted_min(prop, list(element_counts.values()))
        prop_wmax_diff = weighted_max_diff(prop, list(element_counts.values()))
        prop_wmin_diff = weighted_min_diff(prop, list(element_counts.values()))
        prop_wstd = weighted_std(prop, list(element_counts.values()))
  
        results.extend([prop_avg, prop_wavg, prop_max, prop_wmax, prop_min, prop_wmin, prop_max_diff, prop_wmax_diff,
                        prop_min_diff, prop_wmin_diff, prop_std, prop_wstd])
    
    composition = structure.composition
    target_elements = get_target_elements()
    inclusion_feature_list = [1 if el in composition else 0 for el in target_elements]
    fraction_feature_list = [composition.get_atomic_fraction(el) if el in composition else 0.0 for el in target_elements]
    results = fraction_feature_list + results
    results = inclusion_feature_list + results
    
    fractions = np.array(list(element_counts.values())) / total_atoms
    # L2 norm : (sum_i (x_i)^2)^(1/2)
    L2_norm = np.linalg.norm(fractions, ord=2)
    # L3 norm : (sum_i (x_i)^3)^(1/3)
    L3_norm = (np.sum(fractions ** 3)) ** (1/3)
    # Stoichiometry entropy: - sum_i (x_i * log(x_i))
    stoich_entropy = -np.sum(fractions * np.log(fractions))
    species_num = len(element_counts)
    sites_num = total_atoms
    
    results.insert(0, stoich_entropy)
    results.insert(0, L3_norm)
    results.insert(0, L2_norm)
    results.insert(0, species_num)
    
    cif_reduced_formula = structure.composition.reduced_formula
    
    doc = None
    spacegroup_number = None
    crystal_systems = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'trigonal', 'hexagonal', 'cubic']
    #crystal_system = spa.get_crystal_system()
    #one_hot = [1 if cs == crystal_system else 0 for cs in crystal_systems]

    if "root_dir_Eh0.1_FM_ICSD" in cif_file:
        # mp-id 추출
        cif_id = os.path.splitext(os.path.basename(cif_file))[0]
        mp_id = f"mp-{cif_id}"
    
        # CIF의 reduced_formula
        cif_reduced_formula = structure.composition.reduced_formula
    
        # MPRester 검색 및 reduced_formula 비교
        doc = None
        with MPRester('YOUR_MP_API_KEY') as mpr:
            docs = mpr.summary.search(material_ids=[mp_id], fields=["formula_pretty", "symmetry"])
            for d in docs:
                mp_reduced_formula = Composition(d.formula_pretty).reduced_formula
                if mp_reduced_formula.replace(" ", "").lower() == cif_reduced_formula.replace(" ", "").lower():
                    doc = d
                    break
    
        if doc is not None and hasattr(doc, "symmetry"):
            try:
                crystal_system = str(list(doc.symmetry)[0][1]).lower()
            except Exception as e:
                print(f"[ERROR] crystal_system 추출 실패: {e}")
                return None
            try:
                spacegroup_number = list(doc.symmetry)[2][1]
            except Exception as e:
                print(f"[ERROR] spacegroup_number 추출 실패: {e}")
                return None
        else:
            print("[ERROR] doc가 None이거나 symmetry가 없습니다.")
            return None
    
        one_hot = [1 if cs == crystal_system else 0 for cs in crystal_systems]


    else:
        # mp_id가 없으므로 직접 분석
        try:
            spa = SpacegroupAnalyzer(structure)
            crystal_system = spa.get_crystal_system().lower()
            spacegroup_number = spa.get_space_group_number()
        except Exception:
            crystal_system = ""
            spacegroup_number = None

        one_hot = [1 if cs == crystal_system else 0 for cs in crystal_systems]

    spa = SpacegroupAnalyzer(structure)
    conventional_structure = spa.get_conventional_standard_structure()
    
    lattice = conventional_structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    angle_dev = (abs(alpha - 90) + abs(beta - 90) + abs(gamma - 90)) / 270
    
    results.extend(one_hot)
    #results.extend([structure.get_space_group_info()[1]])
    results.extend([spacegroup_number])
    results.extend([conventional_structure.volume])
    #results.extend([species_num])
    #results.extend([sites_num])
    results.extend([calculate_ratios(a, b, c)[0]])
    results.extend([calculate_ratios(a, b, c)[1]])
    results.extend([lattice_cv(a, b, c)])
    results.extend([anisotropy_factor(a, b, c)])
    results.extend([calculate_ratios(alpha, beta, gamma)[0]])
    results.extend([calculate_ratios(alpha, beta, gamma)[1]])
    results.extend([angle_dev])
    
    sine_matrix = SineMatrix(n_atoms_max=192)
    _a = AseAtomsAdaptor.get_atoms(structure)
    atoms = _DSystem(numbers=_a.get_atomic_numbers(), positions=_a.get_positions(),
                     cell=_a.get_cell(), pbc=True)
    sine_mat = sine_matrix.create(atoms)
    sine_mat = sine_matrix.unflatten(sine_mat)
    eigenvalues = np.linalg.eigvalsh(sine_mat)
    results.extend(eigenvalues.tolist())

    return results
    
#%%

import subprocess
import sys
import numpy as np
import pandas as pd
import os
import shutil

import tempfile

def analyze_cif_properties_with_embedding(cif_file, modelpath, predict_script, atom_init_json_path):
    # 기존 특성 벡터 계산
    results = analyze_cif_properties(cif_file)

    # 절대 경로로 고정 (subprocess cwd 변경에 대비)
    modelpath_abs = os.path.abspath(modelpath)
    predict_script_abs = os.path.abspath(predict_script)
    atom_init_abs = os.path.abspath(atom_init_json_path)
    cif_file_abs = os.path.abspath(cif_file)

    # OneDrive 동기화 폴더 밖(시스템 temp)에 고유한 임시 디렉토리 생성 -> WinError 5 회피
    tmp_dir = tempfile.mkdtemp(prefix='cgcnn_emb_')
    try:
        cif_basename = os.path.basename(cif_file)
        shutil.copy(cif_file_abs, tmp_dir)
        shutil.copy(atom_init_abs, tmp_dir)

        # id_prop.csv 파일 생성
        cif_id = cif_basename.replace('.cif', '')
        id_prop_path = os.path.join(tmp_dir, 'id_prop.csv')
        with open(id_prop_path, 'w') as f:
            f.write(f"{cif_id},0\n")

        # predict_hidden.py 호출 (cwd=tmp_dir -> test_results.csv 가 tmp_dir 안에 생성됨)
        cmd = [
            sys.executable, predict_script_abs,
            modelpath_abs, '.',
            '--batch-size', '1',
            '--disable-cuda'
        ]
        subprocess.run(cmd, check=True, cwd=tmp_dir)

        # 임베딩 결과 CSV 로드 (tmp_dir 안)
        embedding_csv = os.path.join(tmp_dir, 'test_results.csv')
        embedding_df = pd.read_csv(embedding_csv, header=None)

        # 임베딩 벡터 추출
        embedding_str = embedding_df.iloc[0, 2]
        embedding_str_clean = embedding_str.replace('tensor(', '').replace(')', '')
        embedding_vector = np.fromstring(embedding_str_clean.strip('[]'), sep=',')

        # 임베딩 벡터 추가
        extended_results = results + embedding_vector.tolist()
    finally:
        # 임시 폴더 안전하게 삭제
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return extended_results

modelpath='pre-trained/formation-energy-per-atom.pth.tar'
predict_script = 'predict_hidden.py'
atom_init_json_path = 'atom_init.json'
#%% ML input feature 생성

import os
import pandas as pd
from tqdm import tqdm

modelpath = 'pre-trained/formation-energy-per-atom.pth.tar'
predict_script = 'predict_hidden.py'
atom_init_json_path = 'atom_init.json'

def process_from_input_csv(input_csv_path, root_dir_1, root_dir_2, output_csv_path, mode):
    df = pd.read_csv(input_csv_path, header=None)
    data = []
    target_elements = get_target_elements()

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing entries"):
        identifier = str(row[0])

        # target_value 선택
        if mode == 'l':
            target_value = row[6]
        else:  # mode == 'u'
            target_value = 0

        # CIF 경로 결정
        if identifier.startswith("mp-"):
            file_id = identifier.replace("mp-", "")
            cif_file = os.path.join(root_dir_1, f"{file_id}.cif")
        else:
            #identifier_no_json = identifier.replace(".json", "")
            #cif_file = os.path.join(root_dir_2, f"{identifier_no_json}.cif")
            
            cif_file = os.path.join(root_dir_3, f"{identifier}.cif")

        # CIF 파일 존재 여부 확인 및 처리
        if os.path.exists(cif_file):
            try:
                features = analyze_cif_properties_with_embedding(cif_file, modelpath, predict_script, atom_init_json_path)
                row_data = [identifier] + features + [target_value]
                data.append(row_data)
            except Exception as e:
                print(f"⚠️  {identifier} 처리 실패: {e}")
        else:
            print(f"⚠️  {identifier} 에 해당하는 CIF 파일이 존재하지 않음")

    # 결과 저장
    if data:
        num_features = len(data[0]) - 2  # ID + features + target
        properties = ['s', 'p', 'd', 'f', 'unfilled_s', 'unfilled_p', 'unfilled_d', 'unfilled_f',
                      'unpaired_elec', 'spin_only_magmom', 'r', 'EN', 'atomic_num', 'mendeleev_num', 'mass', 'E_i', 'EA', 
                      'dipole_polarizability', 'density', 'Tb', 'Tm', 'group', 'period', 'atomic_vol']
        statistics = ['avg', 'wavg', 'max', 'wmax', 'min', 'wmin', 'maxdiff', 'wmaxdiff', 'mindiff', 'wmindiff', 'std', 'wstd']
        columns = ['ID'] + ['species_num'] + ['L2_norm'] + ['L3_norm'] + ['stoi_entropy'] \
                  + [f"{el}_inclusion" for el in target_elements] + [f"{el}_fraction" for el in target_elements] \
                  + [f"{prop}_{stat}" for prop in properties for stat in statistics] \
                  + ['crystal_sys'] * 7 + ['space_group_num'] + ['cell_volume'] + ['lattice_ratio_arith'] + ['lattice_ratio_geo'] \
                  + ['CV'] + ['aniso_factor'] + ['angle_ratio_arith'] + ['angle_ratio_geo'] + ['angle_dev'] \
                  + ['sm_eigenvalue'] * 192 + ['cgcnn_embedding'] * 32 + ['reference']
        result_df = pd.DataFrame(data, columns=columns)
        result_df.to_csv(output_csv_path, index=False)
        print(f"✅ 총 {len(result_df)}개 CIF 처리 완료. 결과 저장: {output_csv_path}")
    else:
        print("❌ 처리된 CIF가 없습니다. 결과 파일이 생성되지 않았습니다.")

# Non-interactive round2 driver: python round2_featurize.py <input_csv> <output_csv> <root_dir_3>
if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    root_dir_3 = sys.argv[3]
    mode = 'u'
    root_dir_1 = 'root_dir_Eh0.1_FM_ICSD'
    root_dir_2 = 'root_dir_novamag'
    process_from_input_csv(input_csv, root_dir_1, root_dir_2, output_csv, mode)
