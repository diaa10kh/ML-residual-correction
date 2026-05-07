from __future__ import print_function
import json
import os
import sys
import traceback


def _script_dir():
    for arg in sys.argv:
        try:
            if os.path.basename(arg).lower() == '00_3d_cpt_deek.py':
                return os.path.dirname(os.path.abspath(arg))
        except Exception:
            pass
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


SCRIPT_DIR = _script_dir()


def _arg_value(flag):
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def _load_case_config():
    path = os.environ.get('MGS_CASE_CONFIG') or _arg_value('--case-config')
    if not path:
        return {}
    with open(path, 'r') as handle:
        config = json.load(handle)
    config['_config_path'] = os.path.abspath(path)
    return config


MGS_CASE_CONFIG = _load_case_config()
MGS_RUN_ID = MGS_CASE_CONFIG.get('run_id', '')
MGS_RUN_DIR = MGS_CASE_CONFIG.get('run_dir', '')
MGS_DEBUG_LOG = os.path.join(MGS_RUN_DIR or os.getcwd(), 'mgs_generator_debug.log')


def _debug(message):
    try:
        folder = os.path.dirname(MGS_DEBUG_LOG)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        with open(MGS_DEBUG_LOG, 'a') as handle:
            handle.write(str(message) + '\n')
    except Exception:
        pass


def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
    try:
        folder = os.path.dirname(MGS_DEBUG_LOG)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        with open(MGS_DEBUG_LOG, 'a') as handle:
            handle.write('UNCAUGHT EXCEPTION\n')
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=handle)
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = _log_uncaught_exception
_debug('Starting 00_3D_CPT_deek.py')
_debug('sys.argv = %s' % sys.argv)
_debug('MGS_CASE_CONFIG env = %s' % os.environ.get('MGS_CASE_CONFIG', ''))
_debug('Loaded config path = %s' % MGS_CASE_CONFIG.get('_config_path', ''))
_debug('Run ID = %s' % MGS_RUN_ID)

Mdb()

if MGS_CASE_CONFIG:
    if not MGS_RUN_DIR:
        MGS_RUN_DIR = os.path.join(SCRIPT_DIR, 'mgs_rigid_correction', 'runs', MGS_RUN_ID)
    if not os.path.isdir(MGS_RUN_DIR):
        os.makedirs(MGS_RUN_DIR)
    os.chdir(MGS_RUN_DIR)
    _debug('Changed directory to run dir: %s' % os.getcwd())
else:
    legacy_dir = r"D:\Publikationen\2024-03-MassScaling_Kouplung\ML residual correction\Numerical model"
    if os.path.isdir(legacy_dir):
        os.chdir(legacy_dir)
        _debug('No case config detected; changed directory to legacy model dir: %s' % os.getcwd())
    else:
        message = (
            'No MGS case config was supplied. Run this file through '
            'mgs_rigid_correction/scripts/02_generate_abaqus_inputs.py, or set '
            'the MGS_CASE_CONFIG environment variable to a runs/.../case_config.json file.'
        )
        _debug(message)
        raise RuntimeError(message)


from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from abaqusConstants import *
import mesh
import string
import numpy as np

abapys_dir = r'C:\Users\cda6556\Desktop\abapys\\'; 

sys.path.insert(0, abapys_dir);
from abapys import Boden, Quader, Zylinder, Linienzug, Linie, KreisbogenPunkte
from abapys import MaterialUndBodensectionErstellen, BodenmaterialUndSectionErstellen
from abapys import abapys_tol, InitialisiereAbapys, Einheitsvektor, Log, BedingteAuswahl, ZweifachbedingteKantenAuswahl
from abapys import Knotentransformation, Zustandsuebertragung
InitialisiereAbapys(session=session, version=version, pfad=abapys_dir);

#
# ----------------------------------------------------
# ----------------------------------------------------
# Geometrien
# ----------------------------------------------------
# ----------------------------------------------------
#  


### Hier kann man zwischen halb- und kompletem Querschnitt auswaehlen 
Querschnitt = 'Viertel'  #: 'Viertel', 'Halb' oder 'Voll'

 
''' Boring View
         
          PfahlRadius*2 
             _|_|_     
              | |                   
               _               
              (_)   
              | | 
              | | 
              | | 
              | | 
              | | 
              | | 
              | | 
              |_|  
         

          ModelBreite*2
      _|________________|_
       |                |
       .  .          .  .
       .  .          .  .
       |InnererRadius*2 |
       |    _|____|_    |
       |     |    |     |
           ..------..
         ..          ..                 
        /     .-_-.    \                
       |     ( (_) )    |   _______________|_
       |\    |'--'|    /|                  |
       | '- '+----+' -' |                  |
       |     +----+     |                  |
       |     |    |     |                  |
       |     +----+     |                  |
       |     |    |     |                  | 
       |     +----+     |                  |
       | .- .+----+. -. |                  |
       |/    |.--.|    \|                  |
       |     (    )     |   ___|_          |  ModelTiefe
       |\    |'--'|    /|      |           |
       | '- '+----+' -' |      |           |
       |     +----+     |      |           |
       |     |    |     |      |           |
       |     |    |     |      |           |
       |     |    |     |      | Height01  |
       |     +----+.    |      |           |
       | .- .+----+. -. |      |           |
       |/    |.--.|    \|      |           |
       |     (    )     |   ___|___________|_
        \     '--'     /       |           |
         ''          ''
          ''------ ''     


''' 

	#### Variables #####
	
ModelBreite,   ModelTiefe,  PfahlRadius,    PfahlLaenge,  InnererRadius,    Height01,    VoidHoehe     = [   
#[m]              [m]           [m]             [m]           [m]            [m]          [m]      
#-----------|-------------|--------------|-------------|---------------|-------------|-------------|
21.25,              28.0,          0.3,         12,           2.0,            12.5,          1.0     #    
#-----------|-------------|--------------|-------------|---------------|-------------|-------------|
];
                                                                                                                                                     
Step_length = 1

Einpress_Weg = 3
New_Einpress_Weg = 9 # Gaenderter Einpressweg in Fall des Einpressens
Einpress_Zeit = New_Einpress_Weg

DichteHoehe = 4
## Simulationsparameter

# Reibung
kontakt_mit_reibung = True;
reibungskoeffizient = 0.15;

# Userroutine
benoetigtUserroutine = False

# Simulationszeiten und -ausgabefrequenz
secondsperoutput = 0.1; # [s]

# Erdbeschleunigung
g = 9.81;


# Scaling factor
Scal = 1;

# Stoffgesetz

# Stoffgesetz= 'Hypoplastisch'
Stoffgesetz= 'Mohr-Coulomb'
# Stoffgesetz= 'Hypoplastisch'

if MGS_CASE_CONFIG:
    D_m = float(MGS_CASE_CONFIG.get('D_m', PfahlRadius * 2.0))
    L_m = float(MGS_CASE_CONFIG.get('L_m', PfahlLaenge))
    velocity_m_per_s = float(MGS_CASE_CONFIG.get('velocity_m_per_s', 1.0))
    ID_percent = float(MGS_CASE_CONFIG.get('ID_percent', 40.0))
    PfahlRadius = D_m / 2.0
    PfahlLaenge = L_m
    InnererRadius = float(MGS_CASE_CONFIG.get('inner_radius_m', max(2.0, 3.333333333 * D_m)))
    New_Einpress_Weg = float(MGS_CASE_CONFIG.get('penetration_m', 0.9 * L_m))
    Einpress_Zeit = New_Einpress_Weg / max(velocity_m_per_s, 1.0e-12)
    Stoffgesetz = MGS_CASE_CONFIG.get('soil_model', 'Hypoplastisch')
    Scal = float(MGS_CASE_CONFIG.get('S', 1.0))
    secondsperoutput = Einpress_Zeit / max(float(MGS_CASE_CONFIG.get('field_output_points', 100.0)), 1.0)
    history_time_interval = Einpress_Zeit / max(float(MGS_CASE_CONFIG.get('history_output_points', 1000.0)), 1.0)
else:
    history_time_interval = 0.01


#
# ----------------------------------------------------
# ----------------------------------------------------
# Steps
# ----------------------------------------------------
# ----------------------------------------------------
#

if (Stoffgesetz == 'Hypoplastisch'):
    scaleFactor=1.0
else:
   scaleFactor=1.0     


schritt_schwerkraft = [
#timePeriod    scaleFactor   l_B_Viscosity   q_B_Viscosity
#[s]              [-]           [-]              [-]
#------------|-------------|---------------|--------------|
   0.1,        scaleFactor,          0.42,             1.2      #
#------------|-------------|---------------|--------------|
];

schritt_einpressen = [
#timePeriod    scaleFactor   l_B_Viscosity   q_B_Viscosity
#[s]              [-]           [-]              [-]
#------------|-------------|---------------|--------------|
 Einpress_Zeit,        scaleFactor,          0.42,             1.2      #
#------------|-------------|---------------|--------------|
];



    
Dead_weight= 270


# ----------------------------------------------------
# ----------------------------------------------------
# Material Creation
# ----------------------------------------------------
# ----------------------------------------------------
#





# Subroutine
userroutine = 'vumat-hypo-2020-hst.for'; 
numDepVar = 20;

# Stahlbauteile
stahl_dichte = 7.8; # [kN/m^3]
#                   E-Modul   Querdehnz.
#                   E [kPa]   nu [-]
#                 |---------|------------|
stahl_elastisch = [ 210e6,    0.3        ];

# Stahlbauteile
stahl_dichte_weich = 0.1; # [kN/m^3]
#                   E-Modul   Querdehnz.
#                   E [kPa]   nu [-]
#                 |---------|------------|
stahl_elastisch_weich = [ 10000,    0.3        ];


MGS_ID_RATIO = 0.4
if MGS_CASE_CONFIG:
    MGS_ID_RATIO = float(MGS_CASE_CONFIG.get('ID_percent', 40.0)) / 100.0


def _void_ratio_from_relative_density(ID_percent, e_min, e_max):
    d = max(0.0, min(1.0, float(ID_percent) / 100.0))
    a = d * (float(e_max) - float(e_min))
    b = 1.0 + float(e_min)
    return (b * float(e_max) - a) / max(a + b, 1.0e-12)


def _case_bool(name, default=False):
    value = MGS_CASE_CONFIG.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'on')


MGS_VOID_RATIO_SOIL_VOID = 0.615854
MGS_VOID_RATIO_UPPER_LAYER = 0.615854
MGS_VOID_RATIO_DOWN_LAYER = 0.55
if MGS_CASE_CONFIG:
    e_min = float(MGS_CASE_CONFIG.get('void_ratio_e_min', 0.49))
    e_max = float(MGS_CASE_CONFIG.get('void_ratio_e_max', 0.86))
    if _case_bool('use_reference_void_profile', False):
        MGS_VOID_RATIO_SOIL_VOID = float(MGS_CASE_CONFIG.get('void_ratio_soil_void', 0.615854))
        MGS_VOID_RATIO_UPPER_LAYER = float(MGS_CASE_CONFIG.get('void_ratio_upper_layer', 0.615854))
        MGS_VOID_RATIO_DOWN_LAYER = float(MGS_CASE_CONFIG.get('void_ratio_down_layer', 0.55))
    else:
        uniform_void_ratio = float(MGS_CASE_CONFIG.get(
            'void_ratio_uniform',
            _void_ratio_from_relative_density(float(MGS_CASE_CONFIG.get('ID_percent', 40.0)), e_min, e_max)
        ))
        MGS_VOID_RATIO_SOIL_VOID = uniform_void_ratio
        MGS_VOID_RATIO_UPPER_LAYER = uniform_void_ratio
        MGS_VOID_RATIO_DOWN_LAYER = uniform_void_ratio
_debug('Void ratios: Soil_Void=%s, Soil_Upper_Layer=%s, Soil_down_Layer=%s' % (
    MGS_VOID_RATIO_SOIL_VOID, MGS_VOID_RATIO_UPPER_LAYER, MGS_VOID_RATIO_DOWN_LAYER))

materialien_boden = [
#   Abaqus-Bez.   Datenbankname        Parameter-Bez.   Saettigung   Lagerungsd.   Stoffgesetz
#   >''           >''                  ''               [0-1]        [0-1]         >''
# |-------------|--------------------|----------------|------------|-------------|-----------------|
  [ 'HYPO-VW96-Sand',  'Fraction E silica sand', '',            0.0,      MGS_ID_RATIO,        Stoffgesetz  ],
# |-------------|--------------------|----------------|------------|-------------|-----------------|
# Die Lagerungsdichte/Verdichtungsgrad kann bestimmt werden aus
#   a) Anfangsdichte rho_0:     D = (rho_0 - rho_min)/(rho_max - rho_min)
#   b) Anfangsporenzahl e_0:    D = (e_max - e_0)/(e_max - e_min) * (e_min + 1)/(e_0 + 1)
];




schichten,     schichtmaterial,       restmaterial = [
#>[0]          >['']                  >''
#[m]
#------------|----------------------|--------------|
 [0.25, 1.5],  ['HYPO-VW96-Sand'],     'Sand'       #
#------------|----------------------|--------------|
];


#
# ----------------------------------------------------
# ----------------------------------------------------
# Mesh
# ----------------------------------------------------
# ----------------------------------------------------
#

CoreKantenSeed,          HorizintalKantenSeed,       OuterKantenSeed    = [        
#[konstant] oder       [konstant] oder      [konstant] oder          
#[klein, gross]        [klein, gross]       [klein, gross]           
#[m] (innen->aussen)   [m] (innen->aussen)  [m] (innen->aussen)
#--------------------|---------------------|-------------------|
0.12,                     [0.9, 3.0],               2.5         # 
#--------------------|---------------------|-------------------|
];


UntenVertikalSeed,  HorizintalKanten_miniSeed,       VoidKantenSeed    = [        
#[konstant] oder       [konstant] oder      [konstant] oder          
#[klein, gross]        [klein, gross]       [klein, gross]           
#[m] (innen->aussen)   [m] (innen->aussen)  [m] (innen->aussen)
#--------------------|---------------------|-------------------|
[0.5, 2.5],                [0.15, 0.8],           0.14         # 
#--------------------|---------------------|-------------------|
];



    
def include(filename):
    candidate = os.path.join(SCRIPT_DIR, filename)
    if os.path.exists(candidate):
        _debug('Including %s' % candidate)
        execfile(candidate)
    elif os.path.exists(filename): 
        _debug('Including %s' % filename)
        execfile(filename)
    else:
        _debug('Include file not found: %s' % filename)


# 1: Erstellung und erste Tests
if (Querschnitt == 'Viertel'):
    GEOname = '90'


if (Stoffgesetz == 'Hypoplastisch'):
    STOFFname = '_HPM'
elif (Stoffgesetz == 'Mohr-Coulomb'):    
    STOFFname = '_MCM'
    
    




# Updated mapping to include 'Einpressen' operation type
directory_map = {
    'Hypoplastisch': {
        'INPUTS': {
            'Einpressen': r"D:\Publikationen\2024-03-MassScaling_Kouplung\ML residual correction\Numerical model\Viertel\HPM\Einpressen"
        }
    },
    'Mohr-Coulomb': {
        'INPUTS': {
            'Einpressen': r"D:\Publikationen\2024-03-MassScaling_Kouplung\ML residual correction\Numerical model\Viertel\MCM\Einpressen"
        }
    }
}


# Function to change directory based on Stoffgesetz, Querschnitt, and operation type
def change_directory(Stoffgesetz, Querschnitt, operation_type):
    if MGS_CASE_CONFIG:
        if not os.path.isdir(MGS_RUN_DIR):
            os.makedirs(MGS_RUN_DIR)
        os.chdir(MGS_RUN_DIR)
        print("Directory changed to {}".format(MGS_RUN_DIR))
        return
    try:
        path = directory_map[Stoffgesetz][Querschnitt][operation_type]
        os.chdir(path)
        print("Directory changed to {}".format(path))
    except KeyError as e:
        print("Invalid key: {}".format(e))



include('02_Create_3D_CPT_deek_Voll.py')

