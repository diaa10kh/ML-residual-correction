#
# ----------------------------------------------------
## Create Pile 
# ----------------------------------------------------
#

partitionsname = 'Pile';
Zeichnung = mymodel.ConstrainedSketch('profil_' + partitionsname, sheetSize=20.0)
Zeichnung.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(0.0, PfahlRadius))           
partPile = mymodel.Part(name='Pile', dimensionality=THREE_D, type=DEFORMABLE_BODY)
partPile.BaseSolidExtrude(sketch=Zeichnung, depth=PfahlLaenge)
del Zeichnung	

	#### Sets creation #####
partPile = mymodel.parts['Pile']	
partPile.Surface(name = 'Pile_Surface', side1Faces= partPile.faces); 
partPile.Set(name = 'Pile_All', cells= partPile.cells); 



sketchFace = BedingteAuswahl(elemente=partPile.faces, bedingung='(elem.pointOn[0][2] < var[0]-var[1]) and (elem.pointOn[0][2] > var[1])', 
   var=[PfahlLaenge, abapys_tol]); 
partPile.Surface(name = 'Mantel', side1Faces= sketchFace); 


partPile.Surface(name = 'Pile_Surface', side1Faces= partPile.faces); 



sketchFace = BedingteAuswahl(elemente=partPile.faces, bedingung='(elem.pointOn[0][2] > var[0]-var[1])', 
   var=[PfahlLaenge, abapys_tol]);
partPile.Set(name = 'Force_Application', faces= sketchFace);    

spitzendruck = BedingteAuswahl(elemente=partPile.faces, bedingung='(elem.pointOn[0][2] < var[0])', 
   var=[abapys_tol]);
partPile.Set(name = 'spitzendruck', faces= spitzendruck);    
partPile.Surface(name = 'spitzendruck', side1Faces= spitzendruck); 

  
RefCoordinates_Pile = (0.,0.,PfahlLaenge)
partPile.ReferencePoint(point=(RefCoordinates_Pile))



    ### create the cuts in the Pile#####


for offset in np.arange(Step_length, PfahlLaenge, Step_length):
    XYDatum1 = partPile.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=offset)
    XYDatum1_cut = partPile.PartitionCellByDatumPlane(datumPlane=partPile.datums[XYDatum1.id], cells=partPile.cells)

    
i = 0
for x in np.arange(0.0, PfahlLaenge+Step_length, Step_length):
    CylindersFaces = BedingteAuswahl(elemente=partPile.faces, bedingung='(elem.pointOn[0][2] > var[0]-var[1]) and (elem.pointOn[0][2] < var[0]+var[1])', 
       var=[x, abapys_tol]);
    name = 'CylindersFaces' + str(i)  # Use the counter variable i for the name
    partPile.Set(name = name, faces= CylindersFaces)
    i += 1  # Increment the counter variable

# create Datum planes cuts


XZDatum = partPile.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0)	   
partPile.PartitionCellByDatumPlane(datumPlane=partPile.datums[XZDatum.id], cells=partPile.cells)      
YZDatum = partPile.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=0)	 
partPile.PartitionCellByDatumPlane(datumPlane=partPile.datums[YZDatum.id], cells=partPile.cells)   

	#### Sets creation #####

rpid = partPile.features['RP'].id;
partPile.Set(name='Pile_RP', referencePoints=(partPile.referencePoints[rpid], ));


#
# ----------------------------------------------------
## Create Soil 
# ----------------------------------------------------
#

partitionsname = 'Soil';
Zeichnung = mymodel.ConstrainedSketch('profil_' + partitionsname, sheetSize=200.0)

if (Querschnitt == 'Viertel'):
    Zeichnung.ArcByCenterEnds(center=(0.0, 0.0), point1=(0.0, ModelBreite), point2=(ModelBreite, 0.0), 
    direction=CLOCKWISE)
    Zeichnung.Line(point1=(ModelBreite, 0.0), point2=(0.0, 0.0))
    Zeichnung.Line(point1=(0.0, ModelBreite), point2=(0.0, 0.0))
        
if (Querschnitt == 'Halb'):
    Zeichnung.ArcByCenterEnds(center=(0.0, 0.0), point1=(ModelBreite, 0.0), point2=(-ModelBreite, 0.0), 
        direction=COUNTERCLOCKWISE)
    Zeichnung.Line(point1=(ModelBreite, 0.0), point2=(-ModelBreite, 0.0))
if (Querschnitt == 'Voll'):
    Zeichnung.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(ModelBreite, 0.0))     
partSoil = mymodel.Part(name='Soil', dimensionality=THREE_D, type=EULERIAN)
partSoil = mymodel.parts['Soil']
partSoil.BaseSolidExtrude(sketch=Zeichnung, depth=ModelTiefe)
del Zeichnung



    ### create the cuts in the soil#####

  
  
## cut (Core)	
partitionsname = 'Core';
sketchFace = BedingteAuswahl(elemente=partSoil.faces, bedingung='(elem.pointOn[0][2] > var[0]-var[1])', 
   var=[ModelTiefe, abapys_tol]);
partSoil.Set(name = 'sketchFace', faces= sketchFace);
if (Querschnitt == 'Viertel'):
    sketchUpEdge =  BedingteAuswahl(elemente=partSoil.edges, bedingung='(elem.pointOn[0][2] > var[0]-var[2]) and (elem.pointOn[0][1] > var[2])', 
        var=[ModelTiefe,ModelBreite, abapys_tol]);   
if (Querschnitt == 'Halb'):
    sketchUpEdge =  BedingteAuswahl(elemente=partSoil.edges, bedingung='(elem.pointOn[0][2] > var[0]-var[2]) and (elem.pointOn[0][1] > var[2])', 
        var=[ModelTiefe,ModelBreite, abapys_tol]);
if (Querschnitt == 'Voll'):
    sketchUpEdge =  BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt((elem.pointOn[0][0]**2) + (elem.pointOn[0][1]**2)) > var[1]-var[2]) and (elem.pointOn[0][2] > var[0]-var[2])', 
        var=[ModelTiefe,ModelBreite, abapys_tol]);        
partSoil.Set(name = 'sketchUpEdge', edges= sketchUpEdge);    
Zeichnung = mymodel.ConstrainedSketch('profil_' + partitionsname, sheetSize=50.66, gridSpacing=5, transform=partSoil.MakeSketchTransform(sketchPlane=sketchFace[0],
   sketchPlaneSide=SIDE1, sketchUpEdge=sketchUpEdge[0],
   sketchOrientation=RIGHT, origin=(0.0, 0.0, ModelTiefe)));
Zeichnung = mymodel.sketches['profil_' + partitionsname];	
partSoil.projectReferencesOntoSketch(sketch=Zeichnung, filter=COPLANAR_EDGES)
Zeichnung.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(0.0, InnererRadius)) 
neupartition = partSoil.PartitionFaceBySketch(sketchUpEdge=sketchUpEdge[0], faces=sketchFace[0], sketch=Zeichnung)
Zeichnung.unsetPrimaryObject()
del Zeichnung

partSoil.features.changeKey(fromName=neupartition.name, toName=partitionsname);
neueKanten = BedingteAuswahl(elemente=partSoil.edges, bedingung='elem.featureName == \'' + partitionsname + '\'');
zachsenid = partSoil.DatumAxisByPrincipalAxis(principalAxis=ZAXIS)
partSoil.PartitionCellByExtrudeEdge(cells=partSoil.cells,
   edges=(neueKanten), line=partSoil.datums[zachsenid.id], sense=REVERSE);
   

## cut (Core2)	
partitionsname = 'Core2';
sketchFace = BedingteAuswahl(elemente=partSoil.faces, bedingung='(elem.pointOn[0][2] > var[0]-var[1])', 
   var=[ModelTiefe, abapys_tol]);
partSoil.Set(name = 'sketchFace', faces= sketchFace);
if (Querschnitt == 'Viertel'):
    sketchUpEdge =  BedingteAuswahl(elemente=partSoil.edges, bedingung='(elem.pointOn[0][2] > var[0]-var[2]) and (elem.pointOn[0][1] > var[2])', 
        var=[ModelTiefe,ModelBreite, abapys_tol]);   
if (Querschnitt == 'Halb'):
    sketchUpEdge =  BedingteAuswahl(elemente=partSoil.edges, bedingung='(elem.pointOn[0][2] > var[0]-var[2]) and (elem.pointOn[0][1] > var[2])', 
        var=[ModelTiefe,ModelBreite, abapys_tol]);
if (Querschnitt == 'Voll'):
    sketchUpEdge =  BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt((elem.pointOn[0][0]**2) + (elem.pointOn[0][1]**2)) > var[1]-var[2]) and (elem.pointOn[0][2] > var[0]-var[2])', 
        var=[ModelTiefe,ModelBreite, abapys_tol]);        
partSoil.Set(name = 'sketchUpEdge', edges= sketchUpEdge);    
Zeichnung = mymodel.ConstrainedSketch('profil_' + partitionsname, sheetSize=50.66, gridSpacing=5, transform=partSoil.MakeSketchTransform(sketchPlane=sketchFace[0],
   sketchPlaneSide=SIDE1, sketchUpEdge=sketchUpEdge[0],
   sketchOrientation=RIGHT, origin=(0.0, 0.0, ModelTiefe)));
Zeichnung = mymodel.sketches['profil_' + partitionsname];	
partSoil.projectReferencesOntoSketch(sketch=Zeichnung, filter=COPLANAR_EDGES)
Zeichnung.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(0.0, InnererRadius*3)) 
neupartition = partSoil.PartitionFaceBySketch(sketchUpEdge=sketchUpEdge[0], faces=sketchFace[0], sketch=Zeichnung)
Zeichnung.unsetPrimaryObject()
del Zeichnung

partSoil.features.changeKey(fromName=neupartition.name, toName=partitionsname);
neueKanten = BedingteAuswahl(elemente=partSoil.edges, bedingung='elem.featureName == \'' + partitionsname + '\'');
zachsenid = partSoil.DatumAxisByPrincipalAxis(principalAxis=ZAXIS)
partSoil.PartitionCellByExtrudeEdge(cells=partSoil.cells,
   edges=(neueKanten), line=partSoil.datums[zachsenid.id], sense=REVERSE);
   
   
   
# create Datum planes cuts


XYDatumVoid = partSoil.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=ModelTiefe-VoidHoehe)	   
Void_Schnitt = partSoil.PartitionCellByDatumPlane(datumPlane=partSoil.datums[XYDatumVoid.id], cells=partSoil.cells)

XYDatumFuss_Cut = partSoil.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=Height01)	   
Biased_Schnitt = partSoil.PartitionCellByDatumPlane(datumPlane=partSoil.datums[XYDatumFuss_Cut.id], cells=partSoil.cells)
	

if (Querschnitt == 'Halb') or (Querschnitt == 'Voll'):   
    YZDatum = partSoil.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=0)	 
    partSoil.PartitionCellByDatumPlane(datumPlane=partSoil.datums[YZDatum.id], cells=partSoil.cells)   

if (Querschnitt == 'Voll'):
    XZDatum = partSoil.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0)	 
    partSoil.PartitionCellByDatumPlane(datumPlane=partSoil.datums[XZDatum.id], cells=partSoil.cells)   


XYDatum_Material = partSoil.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=ModelTiefe-VoidHoehe-DichteHoehe)	   
Biased_Schnitt = partSoil.PartitionCellByDatumPlane(datumPlane=partSoil.datums[XYDatum_Material.id], cells=partSoil.cells)
	



	#### Sets creation #####
partSoil = mymodel.parts['Soil']
partSoil.Set(name = 'Soil_All', cells= partSoil.cells); #: The set 'Soil_All' has been created

Bottom = BedingteAuswahl(elemente=partSoil.faces, bedingung='(elem.pointOn[0][2] < var[0])', 
   var=[abapys_tol]);	
partSoil.Set(name = 'Bottom', faces= Bottom); #: The set 'Bottom' has been created   
#

Back = BedingteAuswahl(elemente=partSoil.faces, bedingung='(sqrt((elem.pointOn[0][0]**2) + (elem.pointOn[0][1]**2)) > (var[0]-var[1]))', 
   var=[ModelBreite, abapys_tol]);	   
partSoil.Set(name = 'Back', faces= Back);  #: The set 'Back' has been created


if (Querschnitt == 'Halb'):
    Front = BedingteAuswahl(elemente=partSoil.faces, bedingung='(elem.pointOn[0][1] < var[0])', 
       var=[abapys_tol]);	
    partSoil.Set(name = 'Front', faces= Front); #: The set 'Front' has been created   
    #


if (Querschnitt == 'Viertel'):
    XZ_Face = BedingteAuswahl(elemente=partSoil.faces, bedingung='(elem.pointOn[0][1] < var[0])', 
       var=[abapys_tol]);	
    partSoil.Set(name = 'XZ_Face', faces= XZ_Face); 
    YZ_Face = BedingteAuswahl(elemente=partSoil.faces, bedingung='(elem.pointOn[0][0] < var[0])', 
       var=[abapys_tol]);	
    partSoil.Set(name = 'YZ_Face', faces= YZ_Face);     
    
Soil_All_Geosta = BedingteAuswahl(elemente=partSoil.cells, bedingung='(elem.pointOn[0][2] < var[0]+var[1])', 
    var=[ModelTiefe - VoidHoehe, abapys_tol]);	
partSoil.Set(name = 'Soil_All_Geosta', cells= Soil_All_Geosta);


Soil_Void = BedingteAuswahl(elemente=partSoil.cells, bedingung='(elem.pointOn[0][2] > var[0]+var[1])', 
    var=[ModelTiefe - VoidHoehe, abapys_tol]);	
partSoil.Set(name = 'Soil_Void', cells= Soil_Void);



Soil_down_Layer = BedingteAuswahl(elemente=partSoil.cells, bedingung='(elem.pointOn[0][2] < var[0]-3+var[1])', 
    var=[ModelTiefe - VoidHoehe, abapys_tol]);	
partSoil.Set(name = 'Soil_down_Layer', cells= Soil_down_Layer);


Soil_Upper_Layer = BedingteAuswahl(elemente=partSoil.cells, bedingung='(elem.pointOn[0][2] < var[0]+var[1]) and (elem.pointOn[0][2] > var[0]-3+var[1])', 
    var=[ModelTiefe - VoidHoehe, abapys_tol]);	
partSoil.Set(name = 'Soil_Upper_Layer', cells= Soil_Upper_Layer);



# #
# # ----------------------------------------------------
# ## Create Stone 
# # ----------------------------------------------------
# #

# partitionsname = 'Stone';
# Zeichnung = mymodel.ConstrainedSketch('profil_' + partitionsname, sheetSize=5.0)
# Zeichnung.rectangle(point1=(-Stone_size, -Stone_size), point2=(Stone_size, Stone_size))
# partStone = mymodel.Part(name='Stone', dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
# partStone.BaseSolidExtrude(sketch=Zeichnung, depth=Stone_size/3)
# del Zeichnung	
# RefCoordinates_Stone = (0.,0.,Stone_size/3)
# partStone.ReferencePoint(point=(RefCoordinates_Stone))

# partStone.RemoveCells(cellList = partStone.cells)

	# #### Sets creation #####

# partStone = mymodel.parts['Stone']	
# rpid = partStone.features['RP'].id;
# partStone.Set(name='Stone_RP', referencePoints=(partStone.referencePoints[rpid], ));
    
# partStone.Set(faces=partStone.faces, name='Stone_All')
# partStone.Surface(name='Stone_All', side1Faces= partStone.faces)



#
# ----------------------------------------------------
## Create soil mesh 
# ----------------------------------------------------
#


# define seed sets for the soil using abapys  

partSoil.setMeshControls(regions=partSoil.cells, technique=SWEEP)

#####################
HorizintalKanten1 = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) > var[0]+var[2]) and (sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) < var[1]-var[2])',
    bedingung2= '(edge.pointOn[0][0] > var[0]+var[2])',
    bedingung3= '(vert1.pointOn[0][0] > vert2.pointOn[0][0])',
    var=[InnererRadius*3, ModelBreite, abapys_tol]);
partSoil.Set(name = 'HorizintalKanten1', edges= HorizintalKanten1);    
partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=HorizintalKanten1[1], 
    end2Edges=HorizintalKanten1[0], minSize=HorizintalKantenSeed[0], maxSize=HorizintalKantenSeed[1], constraint=FINER)		
 
if (Querschnitt == 'Voll') or (Querschnitt == 'Halb'):  
 
    HorizintalKanten2 = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) > var[0]+var[2]) and (sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) < var[1]-var[2])',
        bedingung2= '(edge.pointOn[0][0] < -var[0]-var[2])',
        bedingung3= '(vert1.pointOn[0][0] > vert2.pointOn[0][0])',
        var=[InnererRadius*3, ModelBreite, abapys_tol]);
    partSoil.Set(name = 'HorizintalKanten2', edges= HorizintalKanten2);    
    partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=HorizintalKanten2[1], 
        end2Edges=HorizintalKanten2[0], minSize=HorizintalKantenSeed[0], maxSize=HorizintalKantenSeed[1], constraint=FINER)		
     

HorizintalKanten3 = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) > var[0]+var[2]) and (sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) < var[1]-var[2])',
    bedingung2= '(edge.pointOn[0][1] > var[0]+var[2])',
    bedingung3= '(vert1.pointOn[0][1] > vert2.pointOn[0][1])',
    var=[InnererRadius*3, ModelBreite, abapys_tol]);
partSoil.Set(name = 'HorizintalKanten3', edges= HorizintalKanten3);    
partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=HorizintalKanten3[1], 
    end2Edges=HorizintalKanten3[0], minSize=HorizintalKantenSeed[0], maxSize=HorizintalKantenSeed[1], constraint=FINER)		

	
if (Querschnitt == 'Voll'):    
    HorizintalKanten4 = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) > var[0]+var[2]) and (sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) < var[1]-var[2])',
        bedingung2= '(edge.pointOn[0][1] < -var[0]-var[2])',
        bedingung3= '(vert1.pointOn[0][0] > vert2.pointOn[0][0])',
        var=[InnererRadius*3, ModelBreite, abapys_tol]);
    partSoil.Set(name = 'HorizintalKanten4', edges= HorizintalKanten4);    
    partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=HorizintalKanten4[1], 
        end2Edges=HorizintalKanten4[0], minSize=HorizintalKantenSeed[0], maxSize=HorizintalKantenSeed[1], constraint=FINER)		

##############################
        

CoreKanten = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) < var[0]+var[3]) and (elem.pointOn[0][2] > var[2]-var[3])', 
var=[InnererRadius, ModelTiefe, Height01, abapys_tol]); 	
partSoil.Set(name = 'CoreKanten', edges= CoreKanten);   
partSoil.seedEdgeBySize(edges=CoreKanten, size=CoreKantenSeed, deviationFactor=0.1, constraint=FINER)	



CoreKanten2 = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) < var[0]+var[3]) and (sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) > var[0]-var[3]) and (elem.pointOn[0][2] > var[2]-var[3])', 
var=[InnererRadius*3, ModelTiefe, Height01, abapys_tol]); 	
partSoil.Set(name = 'CoreKanten2', edges= CoreKanten2);   
partSoil.seedEdgeBySize(edges=CoreKanten2, size=1, deviationFactor=0.1, constraint=FINER)	

CoreKanten2_1 = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) < var[0]+var[3]) and (sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) > var[0]-var[3]) and (elem.pointOn[0][2] > var[2]+var[3]) and (elem.pointOn[0][2] < var[1]-var[3]-var[4]) and (elem.pointOn[0][0] < var[3])', 
var=[InnererRadius*3, ModelTiefe, Height01, abapys_tol, VoidHoehe]);
CoreKanten2_2 = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) < var[0]+var[3]) and (sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) > var[0]-var[3]) and (elem.pointOn[0][2] > var[2]+var[3]) and (elem.pointOn[0][2] < var[1]-var[3]-var[4]) and (elem.pointOn[0][1] < var[3])', 
var=[InnererRadius*3, ModelTiefe, Height01, abapys_tol, VoidHoehe]); 	
partSoil.Set(name = 'CoreKanten2', edges= CoreKanten2_1+CoreKanten2_2);   
partSoil.seedEdgeBySize(edges=CoreKanten2_1+CoreKanten2_2, size=0.9, deviationFactor=0.1, constraint=FINER)	


OuterKanten = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) > var[0]-var[1])', 
var=[ModelBreite, abapys_tol]); 	
partSoil.Set(name = 'OuterKanten', edges= OuterKanten);   
partSoil.seedEdgeBySize(edges=OuterKanten, size=OuterKantenSeed, deviationFactor=0.1, constraint=FINER)	


OuterKanten = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) > var[0]-var[1]) and (elem.pointOn[0][1] > var[1]) and (elem.pointOn[0][0] > var[1])', 
var=[ModelBreite, abapys_tol]); 	
partSoil.Set(name = 'OuterKanten', edges= OuterKanten);   
partSoil.seedEdgeBySize(edges=OuterKanten, size=OuterKantenSeed, deviationFactor=0.1, constraint=FINER)	

OuterKanten = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) > var[0]-var[1]) and (elem.pointOn[0][1] > var[1]) and (elem.pointOn[0][0] > var[1])', 
var=[InnererRadius*3, abapys_tol]); 	
partSoil.Set(name = 'OuterKanten', edges= OuterKanten);   
partSoil.seedEdgeBySize(edges=OuterKanten, size=OuterKantenSeed, deviationFactor=0.1, constraint=FINER)	

OuterKanten = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) > var[0]-var[1]) and (elem.pointOn[0][1] > var[1]) and (elem.pointOn[0][0] > var[1])', 
var=[InnererRadius, abapys_tol]); 	
partSoil.Set(name = 'OuterKanten', edges= OuterKanten);   
partSoil.seedEdgeBySize(edges=OuterKanten, size=OuterKantenSeed, deviationFactor=0.1, constraint=FINER)	

# OutHorizintalKante = BedingteAuswahl(elemente=partSoil.edges, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) > var[0]+var[2]) and (sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) < var[1]-var[2]) and (elem.pointOn[0][2] < var[2])', 
    # var=[InnererRadius, ModelBreite, abapys_tol]); 	
# partSoil.Set(name = 'OutHorizintalKante', edges= OutHorizintalKante);   
# partSoil.seedEdgeBySize(edges=OutHorizintalKante, size=OuterKantenSeed, deviationFactor=0.1, constraint=FINER)	


VoidKanten = BedingteAuswahl(elemente=partSoil.edges, bedingung='(elem.pointOn[0][2] < var[0]-var[2]) and (elem.pointOn[0][2] > var[1]+var[2])', 
    var=[ModelTiefe, ModelTiefe-VoidHoehe, abapys_tol]); 	
partSoil.Set(name = 'VoidKanten', edges= VoidKanten);   
partSoil.seedEdgeBySize(edges=VoidKanten, size=VoidKantenSeed, deviationFactor=0.1, constraint=FINER)	


########################




HorizintalKanten_mini1 = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) < 3*var[0]-var[1]) and (sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) > var[0]+var[1]) ',
    bedingung2= '(edge.pointOn[0][0] > var[1])',
    bedingung3= '(vert1.pointOn[0][0] > vert2.pointOn[0][0])',
    var=[InnererRadius, abapys_tol]);
partSoil.Set(name = 'HorizintalKanten_mini1', edges= HorizintalKanten_mini1);    
partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=HorizintalKanten_mini1[1], 
    end2Edges=HorizintalKanten_mini1[0], minSize=HorizintalKanten_miniSeed[0], maxSize=HorizintalKanten_miniSeed[1], constraint=FINER)		

if (Querschnitt == 'Voll') or (Querschnitt == 'Halb'):      
    HorizintalKanten_mini2 = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) < 3*var[0]-var[1]) and (sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) > var[0]+var[1]) ',
        bedingung2= '(edge.pointOn[0][0] < -var[1])',
        bedingung3= '(vert1.pointOn[0][0] > vert2.pointOn[0][0])',
        var=[InnererRadius, abapys_tol]);
    partSoil.Set(name = 'HorizintalKanten_mini2', edges= HorizintalKanten_mini2);    
    partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=HorizintalKanten_mini2[0], 
        end2Edges=HorizintalKanten_mini2[1], minSize=HorizintalKanten_miniSeed[0], maxSize=HorizintalKanten_miniSeed[1], constraint=FINER)		
     

HorizintalKanten_mini3 = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) < 3*var[0]-var[1]) and (sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) > var[0]+var[1]) ',
    bedingung2= '(edge.pointOn[0][1] > var[1])',
    bedingung3= '(vert1.pointOn[0][1] > vert2.pointOn[0][1])',
    var=[InnererRadius, abapys_tol]);
partSoil.Set(name = 'HorizintalKanten_mini3', edges= HorizintalKanten_mini3);    
partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=HorizintalKanten_mini3[1], 
    end2Edges=HorizintalKanten_mini3[0], minSize=HorizintalKanten_miniSeed[0], maxSize=HorizintalKanten_miniSeed[1], constraint=FINER)	
	
if (Querschnitt == 'Voll'):    
    HorizintalKanten_mini4 = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) < 3*var[0]-var[1]) and (sqrt(edge.pointOn[0][0]**2 + edge.pointOn[0][1]**2) > var[0]+var[1]) ',
        bedingung2= '(edge.pointOn[0][1] < -var[1])',
        bedingung3= '(vert1.pointOn[0][0] > vert2.pointOn[0][0])',
        var=[InnererRadius, abapys_tol]);
    partSoil.Set(name = 'HorizintalKanten_mini4', edges= HorizintalKanten_mini4);    
    partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=HorizintalKanten_mini4[1], 
        end2Edges=HorizintalKanten_mini4[0], minSize=HorizintalKanten_miniSeed[0], maxSize=HorizintalKanten_miniSeed[1], constraint=FINER)		
        
        
#######################

UntenVertikal = ZweifachbedingteKantenAuswahl(elemente=partSoil, bedingung1='(edge.pointOn[0][2] > var[1])',
    bedingung2= '(edge.pointOn[0][2] < var[0]-var[1])',
    bedingung3= '(vert1.pointOn[0][2] > vert2.pointOn[0][2])',
    var=[Height01, abapys_tol]);     
partSoil.Set(name = 'UntenVertikal', edges= UntenVertikal); 
partSoil.seedEdgeByBias(biasMethod=SINGLE, end1Edges=UntenVertikal[0], 
    end2Edges=UntenVertikal[1], minSize=UntenVertikalSeed[0], maxSize=UntenVertikalSeed[1], constraint=FINER)   
    
    
CoreKanten_medial = BedingteAuswahl(elemente=partSoil.cells, bedingung='(sqrt(elem.pointOn[0][0]**2 + elem.pointOn[0][1]**2) < var[0]+var[3]) and (elem.pointOn[0][2] < var[2]-var[3])', 
var=[InnererRadius, ModelTiefe, Height01, abapys_tol]); 	
partSoil.Set(name = 'CoreKanten_medial', cells= CoreKanten_medial);   
partSoil.setMeshControls(regions=CoreKanten_medial, technique=SWEEP)





