# McLachlan projected-tangent figure

A self-contained, repository-local Python generator for the approved diagram.
The default SVG includes the three `\dot\theta^\star` velocity labels, the
`-i` factors on both drift labels, and `\bar d`. The projected tangent-map
label remains at the lower left. The gray dashed state guide stays removed.
The black chain still ends at the red target; the residual-arrow revision is
deliberately deferred.

## Run

Place this `mclachlan_figure` directory anywhere inside your repository.
From its parent directory, using Python 3.10 or newer:

```sh
python3 -m pip install -r mclachlan_figure/requirements.txt
python3 mclachlan_figure/generate_mclachlan.py
```

The default output is `mclachlan_figure/outputs/mclachlan_geometry.svg`.
It is resolved relative to the script, so the script also works when invoked
from another working directory. An explicit output path is resolved relative
to your current working directory:

```sh
python3 mclachlan_figure/generate_mclachlan.py \
  --output outputs/figures/mclachlan_geometry.svg
```

To create a PNG preview as well:

```sh
python3 -m pip install -r mclachlan_figure/requirements-png.txt
python3 mclachlan_figure/generate_mclachlan.py --png --scale 3
```

PNG export requires CairoSVG and a working native Cairo runtime. SVG export
uses Matplotlib alone and does not invoke Cairo. The bundled PNG was generated
at scale 2; the script's preview default is scale 3. The SVG has a 1448 by 1086
viewBox and remains vector at any display size.

## Edit

All drawing inputs are near the top of `generate_mclachlan.py`.
`GEOMETRY` contains plane vertices, arrow endpoints, indices, and graphical
scale factors. `COLORS` contains the palette. `LABELS` contains each readable
mathematical expression, font size, color, and explicit baseline position.
For example, `velocity-j` contains:

```python
r"\dot\theta_j^\star\Pi\partial_{\theta_j}|\psi\rangle"
```

The mathematical sources use Matplotlib's MathText TeX-style syntax, with
STIX glyphs exported as outline paths. This is not a full LaTeX compiler.
No LaTeX installation is required, and the SVG has no external font dependency.
No font files, embedded raster images, or original SVG templates are bundled
or needed by the generator. You can use `generate_mclachlan.py` by itself after
installing its dependencies.

Each lower blue arrow is computed from the corresponding black segment:
`blue_tip = origin + blue_scale * (black_end - black_start)`.
Consequently, changing a black endpoint also changes its paired projected
blue arrow. Each dotted guide reuses the upper and lower blue arrow-tip
coordinates. Its dash layout starts and ends with a dash at those endpoints.
The validation code checks parallelism, same direction, and endpoint contact
before export. Guide clipping only keeps dashes from crossing label text.

These coordinates are graphical inputs for a schematic, not output from a
Schrodinger or McLachlan simulation. The velocity stars are mathematical
labels; the generator does not compute or optimize parameter velocities.

`--unstarred` restores the previous unstarred velocity notation. It leaves the
arrow geometry unchanged.

## Tests

```sh
python3 -m unittest discover -s mclachlan_figure -v
```

The tests cover positive parallel scaling, guide-to-arrowhead contact,
zero-residual chain closure, starred/default labels, unstarred compatibility,
self-contained vector export, deterministic SVG generation, and file writing.
The checked-in outputs can be regenerated using the script; the generator
does not read them.

## Provenance

The earlier temporary generator and label-edit scripts did not persist in the
available session files. This script was reconstructed from the saved native
vector `mclachlan_drift_corrected.svg`, preserving its plane and arrow
coordinates, palette, glyph sizes, and label baselines. It also incorporates
the later requested `\dot\theta^\star` change directly in the vector output.
It is a reconstructed source generator, not a recovered copy of the temporary
scripts and not an automatic trace of the generated PNG.
