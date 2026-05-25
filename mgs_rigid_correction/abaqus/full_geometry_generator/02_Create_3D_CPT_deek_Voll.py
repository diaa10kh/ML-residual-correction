
    
# ----------------------------------------------------
# Versionsgeschichte
# ----------------------------------------------------
#
try:
   MGS_CASE_CONFIG
except NameError:
   MGS_CASE_CONFIG = {}

modellversion = 1;
#

modelname = 'CPT_' + str(GEOname) + STOFFname + '_deform' + str(modellversion).zfill(1);
mdb.Model(name=modelname, modelType=STANDARD_EXPLICIT);
mymodel = mdb.models[modelname];

# ----------------------------------------------------
## Create Pile 
# ----------------------------------------------------
#

#
# ----------------------------------------------------
## Create Soil 
# ----------------------------------------------------
#


mesh_script = os.path.join(SCRIPT_DIR, '03_Create_Pile_Soil_Mesh_deek.py')
if os.path.exists(mesh_script):
    execfile(mesh_script)
else:
    execfile('03_Create_Pile_Soil_Mesh_deek.py') 

  

#
# ----------------------------------------------------
## Create Presse 
# ----------------------------------------------------
#

partitionsname = 'Presse';
Zeichnung = mymodel.ConstrainedSketch('profil_' + partitionsname, sheetSize=20.0)
Zeichnung.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(0.0, PfahlRadius))           
partPresse = mymodel.Part(name='Presse', dimensionality=THREE_D, type=DEFORMABLE_BODY)
partPresse.BaseSolidExtrude(sketch=Zeichnung, depth=Step_length)
del Zeichnung	
RefCoordinates_Presse = (0.,0.,Step_length)
partPresse.ReferencePoint(point=(RefCoordinates_Presse))


	#### Sets creation #####
partPresse = mymodel.parts['Presse']	
rpid = partPresse.features['RP'].id;
partPresse.Set(name='Presse_RP', referencePoints=(partPresse.referencePoints[rpid], ));
partPresse.Set(name = 'Presse_All', cells= partPresse.cells); 


Face = BedingteAuswahl(elemente=partPresse.faces, bedingung='(elem.pointOn[0][2] < var[0])', 
   var=[abapys_tol]);
partPresse.Set(name = 'Presse_Unten', faces= Face)
    



# ----------------------------------------------------
# ----------------------------------------------------
# Material Creation
# ----------------------------------------------------
# ----------------------------------------------------
#


    
mymodel.Material(name='Stahl');
mymodel.materials['Stahl'].Density(table=((stahl_dichte, ), ));
mymodel.materials['Stahl'].Elastic(table=((stahl_elastisch[0], stahl_elastisch[1]), ));
mymodel.HomogeneousSolidSection(material='Stahl', name='Pile_Section', thickness=None);
mymodel.HomogeneousSolidSection(material='Stahl', name='Presse_Section', thickness=None);

verwendeteMaterialien = sorted(set(schichtmaterial));

benoetigtUserroutine, verwendeteBodenwerte = BodenmaterialUndSectionErstellen(modell=mymodel,
   verwendeteMaterialien=verwendeteMaterialien, verfuegbareMaterialien=materialien_boden,
   userroutine=userroutine, numDepVar=numDepVar, euler=True);
     

def _apply_mohr_coulomb_case_parameters(model, material_name):
   if (not MGS_CASE_CONFIG) or (Stoffgesetz != 'Mohr-Coulomb'):
      return
   if not MGS_MC_PARAMETERS:
      raise RuntimeError('Mohr-Coulomb case selected without MGS_MC_PARAMETERS')

   material = model.materials[material_name]
   rho = MGS_MC_PARAMETERS['density']
   elastic_E = MGS_MC_PARAMETERS['E']
   elastic_nu = MGS_MC_PARAMETERS['nu']
   phi = MGS_MC_PARAMETERS['phi']
   psi = MGS_MC_PARAMETERS['psi']
   cohesion = MGS_MC_PARAMETERS['cohesion']
   plastic_strain = MGS_MC_PARAMETERS['plastic_strain']

   try:
      material.density.setValues(table=((rho, ), ))
   except Exception:
      material.Density(table=((rho, ), ))

   try:
      material.elastic.setValues(table=((elastic_E, elastic_nu), ))
   except Exception:
      material.Elastic(table=((elastic_E, elastic_nu), ))

   try:
      material.mohrCoulombPlasticity.setValues(table=((phi, psi), ))
   except Exception:
      material.MohrCoulombPlasticity(table=((phi, psi), ))

   material.mohrCoulombPlasticity.MohrCoulombHardening(
      table=((cohesion, plastic_strain), ))

   _debug('Applied Mohr-Coulomb material override to %s' % material_name)


_apply_mohr_coulomb_case_parameters(mymodel, 'HYPO-VW96-Sand')

# Access the density value
density_value = mymodel.materials['HYPO-VW96-Sand'].density.table[0][0]
    

    
# ----------------------------------------------------
# Section Creation and Assignment
# ----------------------------------------------------
	
 
partSoil.SectionAssignment(region=partSoil.sets['Soil_All'], sectionName='secEuler', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)   

partPile.SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
   region=partPile.sets['Pile_All'], sectionName='Pile_Section', thicknessAssignment=FROM_SECTION);

partPresse.SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
   region=partPresse.sets['Presse_All'], sectionName='Presse_Section', thicknessAssignment=FROM_SECTION);
   
#   
# ----------------------------------------------------
# ----------------------------------------------------
# Assembly
# ----------------------------------------------------
# ----------------------------------------------------
#

## load Parts to the Assemply
mymodel.rootAssembly.Instance(name='Pile-1', part=mymodel.parts['Pile'], dependent=ON)
mymodel.rootAssembly.Instance(name='Soil-1', part=mymodel.parts['Soil'], dependent=ON)
mymodel.rootAssembly.Instance(name='Presse-1', part=mymodel.parts['Presse'], dependent=ON)
# mymodel.rootAssembly.Instance(name='Stone-1', part=mymodel.parts['Stone'], dependent=ON)




mymodel.rootAssembly.translate(instanceList=('Pile-1',), vector=(0.0, 0.0, ModelTiefe - VoidHoehe)) #: translate Pile
mymodel.rootAssembly.translate(instanceList=('Presse-1',), vector=(0.0, 0.0, ModelTiefe - VoidHoehe + PfahlLaenge)) #: translate Presse


# mymodel.rootAssembly.translate(instanceList=('Stone-1', ), vector=(0.0, 0.0, ModelTiefe - VoidHoehe - Stone_height)) #: translate Stone


# ----------------------------------------------------
# ----------------------------------------------------
# Steps
# ----------------------------------------------------
# ----------------------------------------------------
#


# ----------------------------------------------------
# Steps: Schwerkraft
# ----------------------------------------------------
mymodel.ExplicitDynamicsStep(name='Schwerkraft', previous='Initial',
   timePeriod=schritt_schwerkraft[0], scaleFactor=schritt_schwerkraft[1],
   linearBulkViscosity=schritt_schwerkraft[2], quadBulkViscosity=schritt_schwerkraft[3]);
#
region = mymodel.rootAssembly.instances['Soil-1'].sets['Soil_All']
mymodel.Gravity(name='Schwerkraft', createStepName='Schwerkraft', 
    comp3=-g, distributionType=UNIFORM, field='', region=region)

# mymodel.Gravity(name='Schwerkraft_Pile', 
    # createStepName='Schwerkraft', comp3=-g, distributionType=UNIFORM, 
    # field='', region=mymodel.rootAssembly.instances['Pile-1'].sets['Pile_All'])
    

    
# ----------------------------------------------------
# Steps: Einpressen
# ----------------------------------------------------
mymodel.ExplicitDynamicsStep(name='Einpressen', previous='Schwerkraft',
   timePeriod=schritt_einpressen[0], scaleFactor=schritt_einpressen[1],
   linearBulkViscosity=schritt_einpressen[2], quadBulkViscosity=schritt_einpressen[3]);



# ----------------------------------------------------
# Step: Field- and Historyoutputrequest
# ----------------------------------------------------
#

# Fieldoutputrequest

if (benoetigtUserroutine):
   userroutineDatei = userroutine;
   variables = ('A', 'U', 'EVF', 'SVAVG', 'SDV');
else:
   userroutineDatei = '';
   variables = ('A', 'U', 'EVF', 'SVAVG');
#


mymodel.fieldOutputRequests.changeKey(fromName='F-Output-1', toName='FieldOutSoil');
mymodel.fieldOutputRequests['FieldOutSoil'].setValues(
    variables=variables, region=mymodel.rootAssembly.allInstances['Soil-1'].sets['Soil_All'], sectionPoints=DEFAULT, timeInterval=secondsperoutput, 
    rebar=EXCLUDE)

       
mymodel.FieldOutputRequest(name='all-profil-vol', 
    createStepName='Schwerkraft', variables=('S', 'U'), 
    timeInterval=secondsperoutput, region=mymodel.rootAssembly.allInstances['Pile-1'].sets['Pile_All'], sectionPoints=DEFAULT, rebar=EXCLUDE)
    
        

# Historyoutputrequest

mymodel.HistoryOutputRequest(name='RF', 
    createStepName='Schwerkraft', variables=('RF1', 'RF2', 'RF3', 'U3'), 
    timeInterval=history_time_interval, region=mymodel.rootAssembly.allInstances['Presse-1'].sets['Presse_RP'], sectionPoints=DEFAULT, 
    rebar=EXCLUDE)


mymodel.HistoryOutputRequest(name='spitzendruck', 
    createStepName='Schwerkraft', variables=('CFN3', ), timeInterval=history_time_interval, 
    region=mymodel.rootAssembly.allInstances['Pile-1'].surfaces['spitzendruck'], sectionPoints=DEFAULT, rebar=EXCLUDE)


mymodel.HistoryOutputRequest(name='mantel', 
    createStepName='Schwerkraft', variables=('CFS3', ), timeInterval=history_time_interval, 
    region=mymodel.rootAssembly.allInstances['Pile-1'].surfaces['Mantel'], sectionPoints=DEFAULT, rebar=EXCLUDE)
    

mymodel.historyOutputRequests.changeKey(
    fromName='H-Output-1', toName='energie')
mymodel.historyOutputRequests['energie'].setValues(
    variables=('ALLIE', 'ALLKE'), timeInterval=history_time_interval)
if (not MGS_CASE_CONFIG) or (not MGS_CASE_CONFIG.get('enable_energy_output', True)):
    mymodel.historyOutputRequests['energie'].suppress()



    
#
# ----------------------------------------------------
# ----------------------------------------------------
# Kontaktbedingungen
# ----------------------------------------------------
# ----------------------------------------------------
#

mymodel.ContactExp(name='Allgemeinkontakt', createStepName='Initial')

mymodel.ContactProperty('Kontakt');
mymodel.interactionProperties['Kontakt'].NormalBehavior(allowSeparation=ON,
   constraintEnforcementMethod=DEFAULT, pressureOverclosure=HARD);
if (kontakt_mit_reibung):
   mymodel.interactionProperties['Kontakt'].TangentialBehavior(
      dependencies=0, directionality=ISOTROPIC, elasticSlipStiffness=None, 
      formulation=PENALTY, fraction=0.005, maximumElasticSlip=FRACTION, 
      pressureDependency=OFF, shearStressLimit=None, slipRateDependency=OFF, 
      table=((reibungskoeffizient, ), ), temperatureDependency=OFF);
else:
   mymodel.interactionProperties['Kontakt'].TangentialBehavior(formulation=FRICTIONLESS);

mymodel.ContactExp(createStepName='Initial', name='Allgemeinkontakt');
mymodel.interactions['Allgemeinkontakt'].includedPairs.setValuesInStep(
   stepName='Initial', useAllstar=ON);
mymodel.interactions['Allgemeinkontakt'].contactPropertyAssignments.appendInStep(
   assignments=((GLOBAL, SELF, 'Kontakt'), ), stepName='Initial');
 

#
# ----------------------------------------------------
# ----------------------------------------------------
# Initiale Bedingungen und Anfangsrandbedingungen
# ----------------------------------------------------
# ----------------------------------------------------
#
mymodel.DisplacementBC(name='Bottom', 
    createStepName='Initial', region=mymodel.rootAssembly.instances['Soil-1'].sets['Bottom'], u1=UNSET, u2=UNSET, u3=SET, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, 
    fieldName='', localCsys=None)
    
    
mymodel.DisplacementBC(name='Back', createStepName='Initial', 
    region=mymodel.rootAssembly.instances['Soil-1'].sets['Back'], u1=SET, u2=SET, u3=UNSET, 
    ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude=UNSET, distributionType=UNIFORM, 
    fieldName='', localCsys=None)
#

if (Querschnitt == 'Halb'):
    mymodel.YsymmBC(name='Front', 
        createStepName='Initial', region=mymodel.rootAssembly.instances['Soil-1'].sets['Front'], localCsys=None)

if (Querschnitt == 'Viertel'):
    mymodel.YsymmBC(name='XZ_Face', 
        createStepName='Initial', region=mymodel.rootAssembly.instances['Soil-1'].sets['XZ_Face'], localCsys=None)
    mymodel.XsymmBC(name='YZ_Face', 
        createStepName='Initial', region=mymodel.rootAssembly.instances['Soil-1'].sets['YZ_Face'], localCsys=None)        
        
#
        
if (Querschnitt == 'Halb'):
    mymodel.DisplacementBC(name='Symmetrie_Pile', 
        createStepName='Initial', region=mymodel.rootAssembly.instances['Pile-1'].sets['Pile_All'], u1=UNSET, u2=SET, u3=UNSET, 
        ur1=SET, ur2=UNSET, ur3=SET, amplitude=UNSET, distributionType=UNIFORM, 
        fieldName='', localCsys=None)

if (Querschnitt == 'Viertel'):
    mymodel.DisplacementBC(name='Symmetrie_Pile', 
        createStepName='Initial', region=mymodel.rootAssembly.instances['Pile-1'].sets['Pile_All'], u1=SET, u2=SET, u3=UNSET, 
        ur1=SET, ur2=SET, ur3=SET, amplitude=UNSET, distributionType=UNIFORM, 
        fieldName='', localCsys=None)
        
 	
    ## Pile BC

    
mymodel.TabularAmplitude(name='Amp_Einpressen', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (Einpress_Zeit, 1.0)))

mymodel.steps['Einpressen'].setValues(improvedDtMethod=ON)

    

mymodel.DisplacementBC(name='Eindringen_Presse', 
    createStepName='Initial', region=mymodel.rootAssembly.instances['Presse-1'].sets['Presse_RP'], u1=SET, u2=SET, u3=SET, ur1=SET, 
    ur2=SET, ur3=SET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
if 'Vorbereiten' in mymodel.steps:
    mymodel.boundaryConditions['Eindringen_Presse'].setValuesInStep(
        stepName='Vorbereiten', u3=FREED)
mymodel.boundaryConditions['Eindringen_Presse'].setValuesInStep(
    stepName='Einpressen', u3=-New_Einpress_Weg, amplitude='Amp_Einpressen')


	## Geostatic stress
region = mymodel.rootAssembly.instances['Soil-1'].sets['Soil_All_Geosta']
mymodel.GeostaticStress(name='K0', region=region, 
    stressMag1=0.0, vCoord1=ModelTiefe-VoidHoehe, stressMag2=-(ModelTiefe-VoidHoehe)*g*density_value, vCoord2=0.0, 
    lateralCoeff1=0.5, lateralCoeff2=None)
    
    

region2=mymodel.rootAssembly.instances['Presse-1'].sets['Presse_All']
region1=mymodel.rootAssembly.instances['Presse-1'].sets['Presse_RP']
mymodel.RigidBody(name='Presse_rigid_body', refPointRegion=region1, bodyRegion=region2)
    

    
# generate the pile mesh

partPile.setMeshControls(regions=partPile.cells, technique=SWEEP)
partPile.seedPart(size=Step_length, deviationFactor=0.1, minSizeFactor=0.1)
partPile.generateMesh()	


# generate the Presse mesh
partPresse.seedPart(size=Step_length, deviationFactor=0.1, minSizeFactor=0.1)
partPresse.generateMesh()	


    
partSoil.generateMesh()	


 
#material assignments
mymodel.MaterialAssignment(name='Materialzuweisung', 
    instanceList=(mymodel.rootAssembly.instances['Soil-1'], ), useFields=False, 
    assignmentList=((mymodel.rootAssembly.instances['Soil-1'].sets['Soil_All_Geosta'], (1, )), ))



#
# ----------------------------------------------------
# ----------------------------------------------------
# Job
# ----------------------------------------------------
# ----------------------------------------------------
#


operation_type = 'Einpressen'  # Can be 'Schlagrammung', 'Vibrationsrammung', or 'Einpressen'
change_directory(Stoffgesetz, Querschnitt, operation_type)
         
        


mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, 
    description='', echoPrint=OFF, explicitPrecision=DOUBLE_PLUS_PACK, 
    historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE, model=modelname, 
    modelPrint=OFF, multiprocessingMode=DEFAULT, name=modelname, 
    nodalOutputPrecision=FULL, numCpus=4, numDomains=4, 
    parallelizationMethodExplicit=DOMAIN, queue=None, resultsFormat=ODB, 
    scratch='', type=ANALYSIS, userSubroutine=userroutineDatei, waitHours=0, waitMinutes=0);
#
myjob = mdb.jobs[modelname];



if (Stoffgesetz == 'Hypoplastisch'):
    # Nachbearbeitung im Schluesselworteditor
    # (letzte Operation vor dem Schreiben der Input-Datei)
    mymodel.keywordBlock.synchVersions(storeNodesAndElements=False);
    for idx, text in enumerate(mymodel.keywordBlock.sieBlocks):
       if (text == '*Contact, op=NEW'):
          mymodel.keywordBlock.replace(idx, '*Contact');

    def GetBlockPosition(model,blockPrefix):
        pos = 0
        for block in model.keywordBlock.sieBlocks:
            if string.lower(block[0:len(blockPrefix)])==string.lower(blockPrefix):
                return pos
            pos=pos+1
        return -1
        
    initial_solution_block = '*Initial Conditions, type=SOLUTION\n\
    Soil-1.Soil_Void, %.12g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n\
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n\
     0.0, 0.0, 0.0, 0.0, 0.0\n\
    Soil-1.Soil_Upper_Layer, %.12g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n\
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n\
     0.0, 0.0, 0.0, 0.0, 0.0\n\
    Soil-1.Soil_down_Layer, %.12g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n\
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0\n\
     0.0, 0.0, 0.0, 0.0, 0.0' % (
        MGS_VOID_RATIO_SOIL_VOID,
        MGS_VOID_RATIO_UPPER_LAYER,
        MGS_VOID_RATIO_DOWN_LAYER,
    )
    mymodel.keywordBlock.synchVersions(storeNodesAndElements=False)
    mymodel.keywordBlock.insert(GetBlockPosition(mymodel,'*Contact')-2, initial_solution_block)




def replace_SDV_with_SDV1(model):
    model.keywordBlock.synchVersions(storeNodesAndElements=False)
    
    # Iterate through each block of text in the keywordBlock
    for idx, block in enumerate(model.keywordBlock.sieBlocks):
        if 'SDV' in block:
            # Replace 'SDV' with 'SDV1' in the block
            new_block = block.replace('SDV', 'SDV1')
            # Replace the old block with the new one
            model.keywordBlock.replace(idx, new_block)

replace_SDV_with_SDV1(mymodel)


def _case_bool(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


if MGS_CASE_CONFIG:
    Scal = float(MGS_CASE_CONFIG.get('S', 1.0))
    if Scal <= 0.0:
        raise ValueError('S must be positive')
    run_job_name = str(MGS_CASE_CONFIG.get('run_id', modelname + '_' + str(int(Scal))))
    num_cpus = int(MGS_CASE_CONFIG.get('num_cpus', 4))
    mymodel.materials['HYPO-VW96-Sand'].density.setValues(table=((density_value * Scal, ), ))
    mymodel.loads['Schwerkraft'].setValues(comp3=-g / Scal, distributionType=UNIFORM, field='')
    mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, 
        description='Full-version geometry MGS run generated from case_config.json', echoPrint=OFF,
        explicitPrecision=DOUBLE_PLUS_PACK, historyPrint=OFF, memory=90,
        memoryUnits=PERCENTAGE, model=modelname, modelPrint=OFF,
        multiprocessingMode=DEFAULT, name=run_job_name, nodalOutputPrecision=FULL,
        numCpus=num_cpus, numDomains=num_cpus, parallelizationMethodExplicit=DOMAIN,
        queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS,
        userSubroutine=userroutineDatei, waitHours=0, waitMinutes=0);
    myjob = mdb.jobs[run_job_name];
    if _case_bool(MGS_CASE_CONFIG.get('save_cae', False)):
        cae_file = MGS_CASE_CONFIG.get('cae_file') or os.path.join(MGS_RUN_DIR, run_job_name + '.cae')
        cae_dir = os.path.dirname(cae_file)
        if cae_dir and not os.path.isdir(cae_dir):
            os.makedirs(cae_dir)
        _debug('Saving CAE file: %s' % cae_file)
        mdb.saveAs(pathName=cae_file)
    myjob.writeInput(consistencyChecking=OFF);
    raise SystemExit(0)

myjob.writeInput(consistencyChecking=OFF);
