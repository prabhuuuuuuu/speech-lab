
import sys
import nltk
import spacy
import webbrowser
from pathlib import Path

from utils import get_text_from_input


# =========================================================
# Load spaCy model
# =========================================================

try:
    nlp = spacy.load("en_core_web_sm")

except OSError:

    print("ERROR: spaCy model 'en_core_web_sm' not found.")
    print()
    print("Install it using:")
    print("python -m spacy download en_core_web_sm")
    print()

    nlp = None


# =========================================================
# Generate Interactive Dependency Tree
# =========================================================

def generate_dependency_tree(doc, output_file="dependency_tree.html"):

    print("Building dependency tree...")

    # -----------------------------------------------------
    # Build parent -> children relationship
    # -----------------------------------------------------

    children = {}

    for token in doc:
        children[token.i] = []

    for token in doc:

        if token.head.i != token.i:
            children[token.head.i].append(token.i)

    # -----------------------------------------------------
    # Find root
    # -----------------------------------------------------

    roots = [
        token.i
        for token in doc
        if token.head.i == token.i
    ]

    if not roots:

        print("ERROR: Could not determine dependency root.")
        return

    root = roots[0]

    # -----------------------------------------------------
    # Sort children
    # -----------------------------------------------------

    for node in children:
        children[node].sort()

    # -----------------------------------------------------
    # Calculate depth
    # -----------------------------------------------------

    depths = {}

    def assign_depth(node, depth):

        depths[node] = depth

        for child in children[node]:
            assign_depth(child, depth + 1)

    assign_depth(root, 0)

    # -----------------------------------------------------
    # Calculate subtree width
    # -----------------------------------------------------

    subtree_width = {}

    def calculate_width(node):

        if not children[node]:

            subtree_width[node] = 1
            return 1

        total = 0

        for child in children[node]:

            total += calculate_width(child)

        subtree_width[node] = max(1, total)

        return subtree_width[node]

    calculate_width(root)

    # -----------------------------------------------------
    # Calculate initial positions
    # -----------------------------------------------------

    horizontal_spacing = 190
    vertical_spacing = 150

    margin_x = 150
    margin_y = 100

    positions = {}

    def assign_positions(node, left_boundary):

        if not children[node]:

            positions[node] = left_boundary + 0.5

            return left_boundary + 1

        current_left = left_boundary

        child_positions = []

        for child in children[node]:

            current_left = assign_positions(
                child,
                current_left
            )

            child_positions.append(
                positions[child]
            )

        positions[node] = (
            min(child_positions) +
            max(child_positions)
        ) / 2

        return current_left

    assign_positions(root, 0)

    # -----------------------------------------------------
    # Canvas dimensions
    # -----------------------------------------------------

    canvas_width = max(
        1400,
        subtree_width[root] *
        horizontal_spacing +
        margin_x * 2
    )

    max_depth = max(depths.values())

    canvas_height = max(
        900,
        (max_depth + 1) *
        vertical_spacing +
        margin_y * 2
    )

    # -----------------------------------------------------
    # Convert to actual coordinates
    # -----------------------------------------------------

    coordinates = {}

    for node, position in positions.items():

        coordinates[node] = {
            "x": margin_x +
                 position *
                 horizontal_spacing,

            "y": margin_y +
                 depths[node] *
                 vertical_spacing
        }

    # -----------------------------------------------------
    # Generate edges
    # -----------------------------------------------------

    edge_svg = ""

    for parent, child_list in children.items():

        for child in child_list:

            edge_svg += f"""
            <line
                id="edge-{parent}-{child}"
                class="edge"
                marker-end="url(#arrow)"
            />

            <text
                id="label-{parent}-{child}"
                class="dependency-label"
            >
                {doc[child].dep_}
            </text>
            """

    # -----------------------------------------------------
    # Generate nodes
    # -----------------------------------------------------

    node_svg = ""

    for node, pos in coordinates.items():

        token = doc[node]

        node_class = "node"

        if node == root:
            node_class += " root-node"

        node_svg += f"""
        <g
            id="node-{node}"
            class="{node_class}"
            data-node="{node}"
            transform="translate({pos['x']},{pos['y']})"
        >

            <circle
                cx="0"
                cy="0"
                r="48"
            />

            <text
                x="0"
                y="-5"
                class="word"
            >
                {token.text}
            </text>

            <text
                x="0"
                y="15"
                class="pos"
            >
                {token.pos_}
            </text>

        </g>
        """

    # =====================================================
    # HTML
    # =====================================================

    html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title></title>

<style>

* {{
    box-sizing: border-box;
}}

html,
body {{
    margin: 0;
    padding: 0;

    width: 100%;
    height: 100%;

    overflow: hidden;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background: #f5f7fa;
}}


/* =====================================================
   HEADER
   ===================================================== */

.header {{

    height: 90px;

    padding: 20px 30px;

    background: white;

    border-bottom:
        1px solid #ddd;

    position: relative;
}}

.header h1 {{

    margin: 0 0 6px 0;

    font-size: 25px;
}}

.header p {{

    margin: 0;

    color: #666;

    font-size: 14px;
}}

.instructions {{

    position: absolute;

    right: 30px;

    top: 25px;

    padding: 10px 15px;

    border-radius: 8px;

    background: #f0f2f5;

    color: #555;

    font-size: 13px;
}}


/* =====================================================
   TREE AREA
   ===================================================== */

.tree-container {{

    width: 100%;

    height:
        calc(100% - 90px);

    overflow: auto;

    padding: 25px;
}}

svg {{

    display: block;

    background: white;

    border:
        1px solid #d5d5d5;

    border-radius: 12px;

    box-shadow:
        0 3px 12px
        rgba(0,0,0,0.08);
}}


/* =====================================================
   EDGES
   ===================================================== */

.edge {{

    stroke: #333;

    stroke-width: 2.5;

    fill: none;

    pointer-events: none;
}}


/* =====================================================
   ARROW
   ===================================================== */

marker {{

    overflow: visible;
}}


/* =====================================================
   DEPENDENCY LABEL
   ===================================================== */

.dependency-label {{

    font-size: 13px;

    font-weight: bold;

    fill: #555;

    text-anchor: middle;

    pointer-events: none;

    paint-order: stroke;

    stroke: white;

    stroke-width: 5px;

    stroke-linejoin: round;
}}


/* =====================================================
   NODES
   ===================================================== */

.node {{

    cursor: grab;
}}

.node:active {{

    cursor: grabbing;
}}

.node circle {{

    fill: white;

    stroke: #222;

    stroke-width: 2.5;
}}

.node:hover circle {{

    fill: #f1f5f9;

    stroke-width: 3.5;
}}


/* =====================================================
   ROOT
   ===================================================== */

.root-node circle {{

    fill: #e8f0ff;

    stroke: #1d4ed8;

    stroke-width: 4;
}}


/* =====================================================
   NODE TEXT
   ===================================================== */

.word {{

    font-size: 15px;

    font-weight: bold;

    text-anchor: middle;

    dominant-baseline: middle;

    pointer-events: none;
}}

.pos {{

    font-size: 11px;

    fill: #666;

    text-anchor: middle;

    pointer-events: none;
}}


/* =====================================================
   SELECTED NODE
   ===================================================== */

.selected circle {{

    stroke: #dc2626;

    stroke-width: 4;
}}

</style>

</head>


<body>


<!-- ===================================================
     HEADER
     =================================================== -->

<div class="header">

    <h1>
        
    </h1>

    <p>



    </p>



</div>


<!-- ===================================================
     TREE
     =================================================== -->

<div class="tree-container">

<svg
    id="tree"
    width="{canvas_width}"
    height="{canvas_height}"
    viewBox="0 0 {canvas_width} {canvas_height}"
>

    <defs>

        <marker
            id="arrow"
            markerWidth="10"
            markerHeight="10"
            refX="8"
            refY="4"
            orient="auto"
            markerUnits="strokeWidth"
        >

            <path
                d="M0,0 L0,8 L9,4 Z"
                fill="#333"
            />

        </marker>

    </defs>


    <!-- EDGES -->

    <g id="edges">

        {edge_svg}

    </g>


    <!-- NODES -->

    <g id="nodes">

        {node_svg}

    </g>

</svg>

</div>


<script>

/* =====================================================
   SVG
   ===================================================== */

const svg =
    document.getElementById("tree");


/* =====================================================
   INITIAL POSITIONS
   ===================================================== */

const initialPositions = {{

"""

    for node, pos in coordinates.items():

        html += f"""
    {node}: {{
        x: {pos['x']},
        y: {pos['y']}
    }},
"""

    html += """
};


/* =====================================================
   CURRENT POSITIONS
   ===================================================== */

const positions = {};

for (
    const [node, position]
    of Object.entries(initialPositions)
) {

    positions[node] = {
        x: position.x,
        y: position.y
    };

}


/* =====================================================
   RELATIONSHIPS
   ===================================================== */

const relationships = [

"""

    for parent, child_list in children.items():

        for child in child_list:

            html += f"""
    {{
        parent: {parent},
        child: {child}
    }},
"""

    html += """
];


/* =====================================================
   GET SVG COORDINATES
   ===================================================== */

function getSVGPoint(event) {

    const point =
        svg.createSVGPoint();

    point.x =
        event.clientX;

    point.y =
        event.clientY;

    const matrix =
        svg
            .getScreenCTM()
            .inverse();

    return point.matrixTransform(matrix);
}


/* =====================================================
   UPDATE EDGES
   ===================================================== */

function updateEdges() {

    relationships.forEach(connection => {

        const parent =
            positions[connection.parent];

        const child =
            positions[connection.child];

        if (!parent || !child) {
            return;
        }


        const dx =
            child.x - parent.x;

        const dy =
            child.y - parent.y;

        const distance =
            Math.sqrt(
                dx * dx +
                dy * dy
            );

        if (distance === 0) {
            return;
        }


        const radius = 48;


        /*
         * Arrow starts at edge of parent circle
         */

        const startX =
            parent.x +
            (dx / distance) *
            radius;

        const startY =
            parent.y +
            (dy / distance) *
            radius;


        /*
         * Arrow ends at edge of child circle
         */

        const endX =
            child.x -
            (dx / distance) *
            radius;

        const endY =
            child.y -
            (dy / distance) *
            radius;


        /*
         * Update arrow
         */

        const edge =
            document.getElementById(
                `edge-${connection.parent}-${connection.child}`
            );

        if (edge) {

            edge.setAttribute(
                "x1",
                startX
            );

            edge.setAttribute(
                "y1",
                startY
            );

            edge.setAttribute(
                "x2",
                endX
            );

            edge.setAttribute(
                "y2",
                endY
            );

        }


        /*
         * Update dependency label
         */

        const label =
            document.getElementById(
                `label-${connection.parent}-${connection.child}`
            );

        if (label) {

            label.setAttribute(
                "x",
                (startX + endX) / 2
            );

            label.setAttribute(
                "y",
                (startY + endY) / 2 - 8
            );

        }

    });

}


/* =====================================================
   DRAGGING STATE
   ===================================================== */

let draggedNode = null;

let dragOffsetX = 0;

let dragOffsetY = 0;


/* =====================================================
   START DRAG
   ===================================================== */

document
    .querySelectorAll(".node")
    .forEach(node => {

        node.addEventListener(
            "pointerdown",
            function(event) {

                event.preventDefault();

                draggedNode =
                    Number(
                        this.dataset.node
                    );


                const point =
                    getSVGPoint(event);


                dragOffsetX =
                    positions[draggedNode].x -
                    point.x;


                dragOffsetY =
                    positions[draggedNode].y -
                    point.y;


                this.setPointerCapture(
                    event.pointerId
                );


                document
                    .querySelectorAll(".node")
                    .forEach(n => {

                        n.classList.remove(
                            "selected"
                        );

                    });


                this.classList.add(
                    "selected"
                );

            }
        );

    });


/* =====================================================
   DRAG
   ===================================================== */

svg.addEventListener(
    "pointermove",
    function(event) {

        if (draggedNode === null) {
            return;
        }


        const point =
            getSVGPoint(event);


        let x =
            point.x +
            dragOffsetX;


        let y =
            point.y +
            dragOffsetY;


        const radius = 48;


        /*
         * Keep node inside canvas
         */

        x = Math.max(
            radius,
            Math.min(
                """ + str(canvas_width) + """ - radius,
                x
            )
        );


        y = Math.max(
            radius,
            Math.min(
                """ + str(canvas_height) + """ - radius,
                y
            )
        );


        positions[draggedNode].x = x;

        positions[draggedNode].y = y;


        const node =
            document.getElementById(
                `node-${draggedNode}`
            );


        if (node) {

            node.setAttribute(
                "transform",
                `translate(${x},${y})`
            );

        }


        updateEdges();

    }
);


/* =====================================================
   END DRAG
   ===================================================== */

svg.addEventListener(
    "pointerup",
    function() {

        draggedNode = null;

    }
);


svg.addEventListener(
    "pointercancel",
    function() {

        draggedNode = null;

    }
);


/* =====================================================
   RESET
   ===================================================== */

svg.addEventListener(
    "dblclick",
    function(event) {

        if (event.target !== svg) {
            return;
        }


        for (
            const [node, position]
            of Object.entries(initialPositions)
        ) {

            positions[node].x =
                position.x;

            positions[node].y =
                position.y;


            const element =
                document.getElementById(
                    `node-${node}`
                );


            if (element) {

                element.setAttribute(
                    "transform",
                    `translate(${position.x},${position.y})`
                );

            }

        }


        document
            .querySelectorAll(".node")
            .forEach(n => {

                n.classList.remove(
                    "selected"
                );

            });


        updateEdges();

    }
);


/* =====================================================
   INITIAL DRAW
   ===================================================== */

updateEdges();

</script>

</body>

</html>
"""

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = Path(output_file)

    output_path.write_text(
        html,
        encoding="utf-8"
    )

    print()
    print("Dependency tree generated successfully.")
    print()
    print("File:")
    print(output_path.absolute())
    print()

    # -----------------------------------------------------
    # Open browser
    # -----------------------------------------------------

    webbrowser.open(
        output_path.absolute().as_uri()
    )

    print("Opened dependency tree in browser.")


# =========================================================
# DEPENDENCY PARSING
# =========================================================

def perform_parsing(text, label="Original"):

    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    print("Text:", text)

    if not text.strip():

        print("ERROR: Empty input.")

        return


    # =====================================================
    # spaCy dependency parsing
    # =====================================================

    if nlp is None:

        print(
            "Dependency parsing skipped because "
            "spaCy model is missing."
        )

        return


    doc = nlp(text)


    print()
    print("Dependency Information")
    print("-" * 60)

    print(
        f"{'WORD':<15}"
        f"{'POS':<10}"
        f"{'DEP':<15}"
        f"{'HEAD':<15}"
    )

    for token in doc:

        print(
            f"{token.text:<15}"
            f"{token.pos_:<10}"
            f"{token.dep_:<15}"
            f"{token.head.text:<15}"
        )


    # =====================================================
    # Generate tree
    # =====================================================

    print()
    print("Generating interactive dependency tree...")

    generate_dependency_tree(
        doc,
        "dependency_tree.html"
    )


    # =====================================================
    # NLTK constituency parsing
    # =====================================================

    try:

        tokens = nltk.word_tokenize(text)

        pos_tags = nltk.pos_tag(tokens)

        grammar = r"""
          NP: {<DT|JJ|NN.*>+}
          PP: {<IN><NP>}
          VP: {<VB.*><NP|PP|CLAUSE>+$}
          CLAUSE: {<NP><VP>}
        """

        cp = nltk.RegexpParser(grammar)

        def format_tree(node):

            if isinstance(node, nltk.Tree):

                return nltk.Tree(
                    node.label(),
                    [
                        format_tree(child)
                        for child in node
                    ]
                )

            return f"{node[0]}({node[1]})"


        tree = cp.parse(pos_tags)


        print()
        print("Constituency Tree")
        print("-" * 60)

        format_tree(tree).pretty_print()


    except Exception as e:

        print(
            "Error during constituency parsing:",
            e
        )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("PARSING PROGRAM")
    print("=" * 60)


    # -----------------------------------------------------
    # Get input
    # -----------------------------------------------------

    if len(sys.argv) > 1:

        input_val = sys.argv[1]

    else:

        input_val = (
            "The inventor built an incredibly fast car."
        )


    print()
    print("Input:", input_val)


    # -----------------------------------------------------
    # Process input through utils.py
    # -----------------------------------------------------

    try:

        text = get_text_from_input(input_val)

    except Exception as e:

        print()
        print("ERROR in get_text_from_input():")
        print(e)

        sys.exit(1)


    # -----------------------------------------------------
    # Run parser
    # -----------------------------------------------------

    if text:

        perform_parsing(
            text,
            label="Original"
        )

    else:

        print()
        print("ERROR: No text was returned from input.")

