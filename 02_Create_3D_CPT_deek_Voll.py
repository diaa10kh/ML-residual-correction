
    
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

modelname = 'CPT_' + str(GEOname) + STOFFname + '_rigid' + str(modellversion).zfill(1);
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


verwendeteMaterialien = sorted(set(schichtmaterial));

benoetigtUserroutine, verwendeteBodenwerte = BodenmaterialUndSectionErstellen(modell=mymodel,
   verwendeteMaterialien=verwendeteMaterialien, verfuegbareMaterialien=materialien_boden,
   userroutine=userroutine, numDepVar=numDepVar, euler=True);
     

# Access the density value
density_value = mymodel.materials['HYPO-VW96-Sand'].density.table[0][0]
    

    
# ----------------------------------------------------
# Section Creation and Assignment
# ----------------------------------------------------
	
 
partSoil.SectionAssignment(region=partSoil.sets['Soil_All'], sectionName='secEuler', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)   

partPile.SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
   region=partPile.sets['Pile_All'], sectionName='Pile_Section', thicknessAssignment=FROM_SECTION);


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

# mymodel.rootAssembly.Instance(name='Stone-1', part=mymodel.parts['Stone'], dependent=ON)




mymodel.rootAssembly.translate(instanceList=('Pile-1',), vector=(0.0, 0.0, ModelTiefe - VoidHoehe)) #: translate Pile

  
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
    timeInterval=history_time_interval, region=mymodel.rootAssembly.allInstances['Pile-1'].sets['Pile_RP'], sectionPoints=DEFAULT, 
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
        

        
 	
    ## Pile BC

    
mymodel.TabularAmplitude(name='Amp_Einpressen', timeSpan=STEP, smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (Einpress_Zeit, 1.0)))

mymodel.steps['Einpressen'].setValues(improvedDtMethod=ON)

    

mymodel.DisplacementBC(name='Eindringen_Pile', 
    createStepName='Initial', region=mymodel.rootAssembly.instances['Pile-1'].sets['Pile_RP'], u1=SET, u2=SET, u3=SET, ur1=SET, 
    ur2=SET, ur3=SET, amplitude=UNSET, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
    

mymodel.boundaryConditions['Eindringen_Pile'].setValuesInStep(
    stepName='Schwerkraft', u3=FREED)  
    
mymodel.boundaryConditions['Eindringen_Pile'].setValuesInStep(
    stepName='Einpressen', u3=-New_Einpress_Weg)


mymodel.boundaryConditions['Eindringen_Pile'].setValuesInStep(
    stepName='Einpressen', amplitude='Amp_Einpressen')





	## Geostatic stress
region = mymodel.rootAssembly.instances['Soil-1'].sets['Soil_All_Geosta']
mymodel.GeostaticStress(name='K0', region=region, 
    stressMag1=0.0, vCoord1=ModelTiefe-VoidHoehe, stressMag2=-(ModelTiefe-VoidHoehe)*g*density_value, vCoord2=0.0, 
    lateralCoeff1=0.5, lateralCoeff2=None)
    
    

    
region2=mymodel.rootAssembly.instances['Pile-1'].sets['Pile_All']
region1=mymodel.rootAssembly.instances['Pile-1'].sets['Pile_RP']
mymodel.RigidBody(name='Pile_rigid_body', refPointRegion=region1, bodyRegion=region2)


    
# generate the pile mesh

partPile.setMeshControls(regions=partPile.cells, technique=SWEEP)
partPile.seedPart(size=0.15, deviationFactor=0.1, minSizeFactor=0.1)
partPile.generateMesh()	



    
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

if MGS_CASE_CONFIG:
    Scal = float(MGS_CASE_CONFIG.get('S', 1.0))
    if Scal <= 0.0:
        raise ValueError('S must be positive')
    run_job_name = str(MGS_CASE_CONFIG.get('run_id', modelname + '_' + str(int(Scal))))
    num_cpus = int(MGS_CASE_CONFIG.get('num_cpus', 4))
    mymodel.materials['HYPO-VW96-Sand'].density.setValues(table=((density_value * Scal, ), ))
    mymodel.loads['Schwerkraft'].setValues(comp3=-g / Scal, distributionType=UNIFORM, field='')
    mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, 
        description='Rigid-pile MGS run generated from case_config.json', echoPrint=OFF,
        explicitPrecision=DOUBLE_PLUS_PACK, historyPrint=OFF, memory=90,
        memoryUnits=PERCENTAGE, model=modelname, modelPrint=OFF,
        multiprocessingMode=DEFAULT, name=run_job_name, nodalOutputPrecision=FULL,
        numCpus=num_cpus, numDomains=num_cpus, parallelizationMethodExplicit=DOMAIN,
        queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS,
        userSubroutine=userroutineDatei, waitHours=0, waitMinutes=0);
    myjob = mdb.jobs[run_job_name];
    myjob.writeInput(consistencyChecking=OFF);
    raise SystemExit(0)

myjob.writeInput(consistencyChecking=OFF);



# 
# ----------------------------------------------------
# Mass scaling Einpressen s = 10 
# ----------------------------------------------------
#

# Scaling factor
Scal = 10;

mymodel.materials['HYPO-VW96-Sand'].density.setValues(table=((density_value*Scal, ), ))
mymodel.loads['Schwerkraft'].setValues(comp3=-g/10, distributionType=UNIFORM, field='')

  

mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, 
    description='', echoPrint=OFF, explicitPrecision=DOUBLE_PLUS_PACK, 
    historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE, model=mymodel, 
    modelPrint=OFF, multiprocessingMode=DEFAULT, name=modelname+'_S10', 
    nodalOutputPrecision=FULL, numCpus=4, numDomains=4, 
    parallelizationMethodExplicit=DOMAIN, queue=None, resultsFormat=ODB, 
    scratch='', type=ANALYSIS, userSubroutine=userroutineDatei, waitHours=0, waitMinutes=0);

myjob = mdb.jobs[modelname+'_S10'];
    
myjob.writeInput(consistencyChecking=OFF);


# 
# ----------------------------------------------------
# Mass scaling Einpressen s = 50
# ----------------------------------------------------
#

# Scaling factor
Scal = 50;

mymodel.materials['HYPO-VW96-Sand'].density.setValues(table=((density_value*Scal, ), ))
mymodel.loads['Schwerkraft'].setValues(comp3=-g/50, distributionType=UNIFORM, field='')

  

mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, 
    description='', echoPrint=OFF, explicitPrecision=DOUBLE_PLUS_PACK, 
    historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE, model=mymodel, 
    modelPrint=OFF, multiprocessingMode=DEFAULT, name=modelname+'_S50', 
    nodalOutputPrecision=FULL, numCpus=4, numDomains=4, 
    parallelizationMethodExplicit=DOMAIN, queue=None, resultsFormat=ODB, 
    scratch='', type=ANALYSIS, userSubroutine=userroutineDatei, waitHours=0, waitMinutes=0);

myjob = mdb.jobs[modelname+'_S50'];
    
myjob.writeInput(consistencyChecking=OFF);


