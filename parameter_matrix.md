# Build123d Parameter Matrix for LoRA Training

This matrix contains the **exact parameters** for build123d methods to prevent AI models from hallucinating non-existent parameters. Each method is documented with all valid parameter combinations.

## 3D Objects (BuildPart)

### Box
```python
# EXACT SIGNATURE: Box(length, width, height, rotation=..., align=..., mode=...)
Box(10, 5, 3)                                           # positional
Box(length=10, width=5, height=3)                       # keyword
Box(10, 5, 3, rotation=(0, 0, 45))                     # with rotation
Box(10, 5, 3, align=(Align.MIN, Align.CENTER, Align.MAX))  # with alignment
Box(10, 5, 3, mode=Mode.ADD)                           # with mode

# DOES NOT EXIST: center, material, thickness, depth
```

### Cylinder
```python
# EXACT SIGNATURE: Cylinder(radius, height, arc_size=360, rotation=..., align=..., mode=...)
Cylinder(radius=10, height=20)                         # keyword form
Cylinder(10, 20)                                       # positional form
Cylinder(radius=10, height=20, arc_size=180)          # partial cylinder
Cylinder(10, 20, rotation=(0, 0, 45))                 # with rotation
Cylinder(10, 20, align=(Align.CENTER, Align.CENTER, Align.MIN))  # with alignment
Cylinder(10, 20, mode=Mode.SUBTRACT)                  # subtraction mode

# DOES NOT EXIST: diameter, center, length, width
```

### Cone
```python
# EXACT SIGNATURE: Cone(bottom_radius, top_radius, height, arc_size=360, rotation=..., align=..., mode=...)
Cone(bottom_radius=10, top_radius=5, height=15)       # keyword form
Cone(10, 5, 15)                                       # positional form
Cone(10, 5, 15, arc_size=270)                        # partial cone
Cone(10, 5, 15, rotation=(0, 0, 30))                 # with rotation

# DOES NOT EXIST: radius, diameter, angle
```

### Sphere
```python
# EXACT SIGNATURE: Sphere(radius, arc_size1=-90, arc_size2=90, arc_size3=360, rotation=..., align=..., mode=...)
Sphere(radius=10)                                     # basic sphere
Sphere(10)                                           # positional form
Sphere(radius=10, arc_size1=-45, arc_size2=45)      # partial sphere
Sphere(10, rotation=(45, 0, 0))                     # with rotation

# DOES NOT EXIST: diameter, center, size
```

### Hole
```python
# EXACT SIGNATURE: Hole(radius, depth=None, mode=Mode.SUBTRACT)
Hole(radius=3)                                       # through hole
Hole(3)                                             # positional form
Hole(radius=3, depth=10)                           # blind hole
Hole(3, 10)                                        # positional depth

# DOES NOT EXIST: diameter, center_x, center_y, thread, size
```

### CounterBoreHole
```python
# EXACT SIGNATURE: CounterBoreHole(radius, counter_bore_radius, counter_bore_depth, depth=None, mode=Mode.SUBTRACT)
CounterBoreHole(radius=3, counter_bore_radius=6, counter_bore_depth=2)
CounterBoreHole(3, 6, 2)                           # positional form
CounterBoreHole(3, 6, 2, depth=15)                 # with depth

# DOES NOT EXIST: cbore_radius, cb_radius, counterbore_radius
```

### CounterSinkHole
```python
# EXACT SIGNATURE: CounterSinkHole(radius, counter_sink_radius, depth=None, counter_sink_angle=82, mode=Mode.SUBTRACT)
CounterSinkHole(radius=3, counter_sink_radius=6)   # basic countersink
CounterSinkHole(3, 6)                              # positional form
CounterSinkHole(3, 6, depth=15)                    # with depth
CounterSinkHole(3, 6, counter_sink_angle=90)       # custom angle

# DOES NOT EXIST: csink_radius, cs_radius, countersink_radius
```

### Torus
```python
# EXACT SIGNATURE: Torus(major_radius, minor_radius, arc_size1=360, arc_size2=360, rotation=..., align=..., mode=...)
Torus(major_radius=20, minor_radius=5)             # basic torus
Torus(20, 5)                                       # positional form
Torus(20, 5, arc_size1=180)                       # partial torus

# DOES NOT EXIST: radius, inner_radius, outer_radius
```

### Wedge
```python
# EXACT SIGNATURE: Wedge(dx, dy, dz, xmin=0, zmin=0, x2min=0, z2min=0, rotation=..., align=..., mode=...)
Wedge(dx=10, dy=8, dz=6)                          # basic wedge
Wedge(10, 8, 6)                                   # positional form
Wedge(10, 8, 6, xmin=2, zmin=1)                  # with offsets

# DOES NOT EXIST: length, width, height, angle
```

## 2D Objects (BuildSketch)

### Circle
```python
# EXACT SIGNATURE: Circle(radius, align=..., mode=...)
Circle(radius=10)                                  # keyword form
Circle(10)                                        # positional form
Circle(10, align=(Align.MIN, Align.CENTER))       # with alignment
Circle(10, mode=Mode.SUBTRACT)                    # subtraction mode

# DOES NOT EXIST: diameter, center, center_x, center_y, size, radius_x, radius_y
```

### Rectangle
```python
# EXACT SIGNATURE: Rectangle(width, height, align=..., mode=...)
Rectangle(width=20, height=15)                    # keyword form
Rectangle(20, 15)                                 # positional form
Rectangle(20, 15, align=(Align.MIN, Align.MAX))   # with alignment
Rectangle(20, 15, mode=Mode.SUBTRACT)             # subtraction mode

# DOES NOT EXIST: length, size, x, y, center, center_x, center_y
```

### RectangleRounded
```python
# EXACT SIGNATURE: RectangleRounded(width, height, radius, align=..., mode=...)
RectangleRounded(width=20, height=15, radius=2)   # keyword form
RectangleRounded(20, 15, 2)                       # positional form
RectangleRounded(20, 15, 2, align=(Align.CENTER, Align.MIN))

# DOES NOT EXIST: corner_radius, fillet_radius, round_radius
```

### RegularPolygon
```python
# EXACT SIGNATURE: RegularPolygon(radius, side_count, major_radius=True, rotation=0, align=..., mode=...)
RegularPolygon(radius=10, side_count=6)           # keyword form
RegularPolygon(10, 6)                             # positional form
RegularPolygon(10, 6, major_radius=False)         # using minor radius
RegularPolygon(10, 6, rotation=30)                # with rotation
RegularPolygon(10, 8, align=(Align.MIN, Align.CENTER))  # with alignment

# DOES NOT EXIST: sides, num_sides, diameter, size, apothem (use major_radius=False)
```

### Ellipse
```python
# EXACT SIGNATURE: Ellipse(x_radius, y_radius, rotation=0, align=..., mode=...)
Ellipse(x_radius=15, y_radius=10)                 # keyword form
Ellipse(15, 10)                                   # positional form
Ellipse(15, 10, rotation=45)                      # with rotation
Ellipse(15, 10, align=(Align.MAX, Align.MIN))     # with alignment

# DOES NOT EXIST: width, height, radius, major_radius, minor_radius
```

### Polygon
```python
# EXACT SIGNATURE: Polygon(*pts, align=..., mode=...)
Polygon((0, 0), (10, 0), (15, 8), (5, 12))       # point sequence
Polygon(*[(0, 0), (10, 0), (15, 8), (5, 12)])    # unpacked list

# DOES NOT EXIST: points, vertices, coords
```

### Triangle
```python
# EXACT SIGNATURE: Triangle(a, b, c, align=..., mode=...)
Triangle(a=10, b=8, c=12)                         # side lengths
Triangle(10, 8, 12)                               # positional form

# DOES NOT EXIST: points, vertices, base, height
```

### Trapezoid
```python
# EXACT SIGNATURE: Trapezoid(width, height, left_side_angle=90, right_side_angle=90, align=..., mode=...)
Trapezoid(width=20, height=10)                    # rectangular trapezoid
Trapezoid(20, 10, left_side_angle=75)            # angled left side
Trapezoid(20, 10, 75, 105)                       # both sides angled

# DOES NOT EXIST: base, top, angle, slope
```

## 1D Objects (BuildLine)

### Line
```python
# EXACT SIGNATURE: Line(start, end, mode=...)
Line(start=(0, 0), end=(10, 5))                  # keyword form
Line((0, 0), (10, 5))                            # positional form

# DOES NOT EXIST: length, angle, direction, from_point, to_point
```

### CenterArc
```python
# EXACT SIGNATURE: CenterArc(center, radius, start_angle, arc_size, mode=...)
CenterArc(center=(0, 0), radius=10, start_angle=0, arc_size=90)
CenterArc((0, 0), 10, 0, 90)                     # positional form

# DOES NOT EXIST: end_angle, sweep_angle, diameter
```

### RadiusArc
```python
# EXACT SIGNATURE: RadiusArc(start, end, radius, short_sagitta=True, mode=...)
RadiusArc(start=(0, 0), end=(10, 0), radius=8)   # keyword form
RadiusArc((0, 0), (10, 0), 8)                    # positional form
RadiusArc((0, 0), (10, 0), 8, short_sagitta=False)  # long arc

# DOES NOT EXIST: center, arc_size, sweep_angle
```

### Spline
```python
# EXACT SIGNATURE: Spline(*points, tangents=None, tangent_scalars=None, periodic=False, parameters=None, scale=True, tol=1e-6, mode=...)
Spline((0, 0), (5, 3), (10, 0))                  # point sequence
Spline((0, 0), (10, 5), tangents=((1, 0), (0, 1)))  # with tangents
Spline(*[(0, 0), (5, 3), (10, 0)])               # unpacked list

# DOES NOT EXIST: control_points, degree, knots, weights (use Bezier for weighted curves)
```

### Polyline
```python
# EXACT SIGNATURE: Polyline(*pts, mode=...)
Polyline((0, 0), (10, 0), (10, 5), (0, 5))       # point sequence
Polyline(*[(0, 0), (10, 0), (10, 5), (0, 5)])    # unpacked list

# DOES NOT EXIST: points, vertices, closed
```

### Bezier
```python
# EXACT SIGNATURE: Bezier(*cntl_pnts, weights=None, mode=...)
Bezier((0, 0), (3, 5), (7, 5), (10, 0))          # control points
Bezier((0, 0), (5, 8), (10, 0), weights=[1, 2, 1])  # weighted

# DOES NOT EXIST: control_points, degree, knots
```

## Operations

### extrude()
```python
# EXACT SIGNATURE: extrude(objects=None, amount=0, dir=None, until=..., both=False, taper=0, clean=True, mode=...)
extrude(amount=10)                                # simple extrude
extrude(amount=10, dir=(0, 0, 1))                # with direction
extrude(amount=10, taper=5)                      # tapered extrude
extrude(amount=10, both=True)                    # both directions
extrude(objects=sketch, amount=10)               # explicit object

# DOES NOT EXIST: height, distance, depth, length
```

### revolve()
```python
# EXACT SIGNATURE: revolve(objects=None, axis=Axis.Z, revolution_arc=360, mode=...)
revolve()                                        # full revolution around Z
revolve(axis=Axis.X)                            # around X axis
revolve(revolution_arc=180)                      # partial revolution
revolve(axis=(Axis((0, 0, 0), (1, 1, 0))))     # custom axis

# DOES NOT EXIST: angle, sweep_angle, degrees, around
```

### loft()
```python
# EXACT SIGNATURE: loft(sections=None, ruled=False, clean=True, mode=...)
loft()                                          # loft pending faces
loft(sections=[face1, face2, face3])           # explicit sections
loft(ruled=True)                               # ruled surface
loft(clean=False)                              # no cleanup

# DOES NOT EXIST: profiles, cross_sections, through
```

### sweep()
```python
# EXACT SIGNATURE: sweep(objects=None, path=None, multisection=False, is_frenet=True, mode=...)
sweep()                                         # sweep pending faces along pending edges
sweep(path=wire)                               # explicit path
sweep(multisection=True)                       # multiple cross-sections
sweep(is_frenet=False)                         # non-Frenet frame

# DOES NOT EXIST: profile, along, guide, trajectory
```

### fillet()
```python
# EXACT SIGNATURE: fillet(objects, radius, length=None, length2=None, mode=...)
fillet(edges, radius=2)                        # constant radius
fillet(edges, radius=2, length=5)              # variable radius
fillet(vertices, radius=3)                     # vertex fillet

# DOES NOT EXIST: corner_radius, round_radius, size
```

### chamfer()
```python
# EXACT SIGNATURE: chamfer(objects, length, length2=None, angle=None, reference=None)
chamfer(edges, length=2)                       # equal chamfer
chamfer(edges, length=2, length2=3)            # unequal chamfer
chamfer(edges, length=2, angle=45)             # length and angle
chamfer(vertices, length=1.5)                  # vertex chamfer

# DOES NOT EXIST: size, distance, bevel, cut
```

### offset()
```python
# EXACT SIGNATURE: offset(objects=None, amount=0, openings=None, kind=Kind.ARC, side=Side.BOTH, closed=True, min_edge_length=None, mode=...)
offset(amount=2)                               # offset pending objects
offset(faces, amount=-1)                       # inward offset
offset(amount=2, openings=[face1, face2])      # with openings
offset(amount=1, kind=Kind.INTERSECTION)       # sharp corners

# DOES NOT EXIST: distance, thickness, inset, outset
```

## Location Contexts

### Locations
```python
# EXACT SIGNATURE: Locations(*pts)
with Locations((10, 0), (0, 10), (-10, 0)):    # multiple points
with Locations(*[(10, 0), (0, 10)]):           # unpacked list

# DOES NOT EXIST: points, positions, coords
```

### GridLocations
```python
# EXACT SIGNATURE: GridLocations(x_spacing, y_spacing, x_count, y_count)
with GridLocations(x_spacing=10, y_spacing=8, x_count=3, y_count=2):
with GridLocations(10, 8, 3, 2):               # positional form

# DOES NOT EXIST: spacing, count, rows, cols, step_x, step_y
```

### PolarLocations
```python
# EXACT SIGNATURE: PolarLocations(radius, count, start_angle=0, angular_range=360, rotate_parts=True)
with PolarLocations(radius=15, count=6):       # basic polar array
with PolarLocations(15, 6):                   # positional form
with PolarLocations(15, 6, start_angle=30):   # rotated start
with PolarLocations(15, 6, angular_range=180): # partial circle
with PolarLocations(15, 6, rotate_parts=False): # don't rotate parts

# DOES NOT EXIST: angle_step, increment, circumference
```

### HexLocations
```python
# EXACT SIGNATURE: HexLocations(radius, x_count, y_count, pointy_top=True)
with HexLocations(radius=10, x_count=5, y_count=4):  # basic hex grid
with HexLocations(10, 5, 4):                   # positional form
with HexLocations(10, 5, 4, pointy_top=False): # flat top orientation

# DOES NOT EXIST: spacing, pitch, step
```

## Selectors

### Basic Selectors
```python
# Available on all builders:
.vertices()                                    # all vertices
.edges()                                      # all edges  
.faces()                                      # all faces (BuildPart/BuildSketch)
.solids()                                     # all solids (BuildPart only)
.wires()                                      # all wires

# DOES NOT EXIST: .nodes(), .surfaces(), .volumes(), .shells()
```

### Filter Methods
```python
# EXACT SIGNATURE: .filter_by(filter_type, tolerance=1e-5)
.edges().filter_by(GeomType.CIRCLE)          # filter by geometry type
.faces().filter_by(Axis.Z)                   # filter by normal direction
.vertices().filter_by(lambda v: v.Z > 5)     # custom filter function

# DOES NOT EXIST: .filter(), .where(), .select()
```

### Sort Methods
```python
# EXACT SIGNATURE: .sort_by(key=Axis.Z, reverse=False)
.faces().sort_by(Axis.Z)                      # sort by Z position
.edges().sort_by(SortBy.LENGTH)               # sort by length
.vertices().sort_by(lambda v: v.X)            # custom sort key
.faces().sort_by(Axis.Z, reverse=True)        # reverse order

# DOES NOT EXIST: .order_by(), .arrange()
```

### Group Methods
```python
# EXACT SIGNATURE: .group_by(axis=Axis.Z, tolerance=1e-5)
.faces().group_by(Axis.Z)                     # group by Z position
.edges().group_by()                           # group by position (all axes)

# DOES NOT EXIST: .cluster(), .partition()
```

## Common Mode Values
```python
Mode.ADD                                      # add to current object (default)
Mode.SUBTRACT                                 # subtract from current object
Mode.INTERSECT                               # intersect with current object
Mode.REPLACE                                  # replace current object

# DOES NOT EXIST: Mode.UNION, Mode.DIFFERENCE, Mode.CUT
```

## Common Align Values
```python
Align.MIN                                     # align to minimum
Align.CENTER                                  # align to center (default)
Align.MAX                                     # align to maximum

# 2D alignment: (Align.CENTER, Align.MIN)
# 3D alignment: (Align.MIN, Align.CENTER, Align.MAX)

# DOES NOT EXIST: Align.LEFT, Align.RIGHT, Align.TOP, Align.BOTTOM
```

## Training Data Patterns

### Pattern 1: Correct Parameter Usage
```python
# CORRECT - Box only has length, width, height
with BuildPart() as part:
    Box(20, 15, 5)
    
# CORRECT - Cylinder only has radius, height
with BuildPart() as part:
    Cylinder(radius=10, height=8)
    
# CORRECT - Hole only has radius, depth (optional)
with BuildPart() as part:
    Box(20, 15, 5)
    Hole(radius=2, depth=10)
```

### Pattern 2: Parameter Combinations
```python
# Show all valid parameter forms for each method
with BuildPart() as part:
    # Box: positional and keyword forms
    Box(10, 8, 3)
    Box(length=10, width=8, height=3)
    
    # Cylinder: with all optional parameters
    Cylinder(radius=5, height=12, arc_size=270, rotation=(0, 0, 45))
    
    # Holes: various depth options
    Hole(3)  # through hole
    Hole(radius=3, depth=8)  # blind hole
```

### Pattern 3: Context-Aware Operations
```python
# Operations that work on pending objects
with BuildPart() as part:
    Box(20, 15, 5)
    # fillet() works on edges, requires radius parameter
    fillet(part.edges().filter_by(Axis.Z), radius=2)
    
    # chamfer() works on edges, requires length parameter  
    chamfer(part.edges().group_by(Axis.Z)[-1], length=1)
```

### Pattern 4: Selector Accuracy
```python
# Correct selector method signatures
with BuildPart() as part:
    Cylinder(radius=10, height=5)
    
    # Correct: sort_by() with Axis parameter
    top_face = part.faces().sort_by(Axis.Z)[-1]
    
    # Correct: filter_by() with GeomType parameter
    circular_edges = part.edges().filter_by(GeomType.CIRCLE)
    
    # Correct: group_by() returns list of lists
    edge_groups = part.edges().group_by(Axis.Z)
```

## Key Points for Training Data

1. **Parameter Names**: Use exact parameter names from the API - never invent new ones
2. **Positional vs Keyword**: Show both forms where applicable
3. **Optional Parameters**: Include examples with and without optional parameters
4. **Mode Parameter**: Most objects accept `mode=Mode.ADD/SUBTRACT/INTERSECT`
5. **Alignment**: 2D objects use 2-tuple, 3D objects use 3-tuple alignment
6. **Context Dependency**: Some operations work on "pending objects" from the current builder context
7. **Selector Chaining**: Methods like `.filter_by().sort_by()` can be chained
8. **Return Types**: Selectors return ShapeList objects that support indexing and slicing

This matrix should significantly reduce parameter hallucination in your LoRA by providing concrete examples of every valid parameter combination.