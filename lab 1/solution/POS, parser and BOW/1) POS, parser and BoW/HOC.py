import sys
import math
import nltk
import spacy
import webbrowser
from pathlib import Path


# =========================================================
# Load spaCy model
# =========================================================

try:
    nlp = spacy.load("en_core_web_sm")

except OSError:
    print("spaCy model 'en_core_web_sm' is missing.")
    print("Install it using:")
    print("python -m spacy download en_core_web_sm")
    nlp = None


# =========================================================
# Modify word order
# =========================================================

def modify_sentence(text):

    words = text.split()

    if len(words) < 4:
        return text

    # Keep first and last words in place.
    # Reverse the words in between.
    first = words[0]
    last = words[-1]

    middle = words[1:-1]

    middle.reverse()

    modified = [first] + middle + [last]

    return " ".join(modified)


# =========================================================
# POS Tagging
# =========================================================

def perform_pos_tagging(text):

    print()
    print("=" * 65)
    print("PART-OF-SPEECH TAGGING")
    print("=" * 65)

    print("Text:", text)

    if nlp is None:
        return

    doc = nlp(text)

    print()

    print(
        f"{'WORD':<18}"
        f"{'POS':<12}"
        f"{'TAG':<12}"
        f"{'LEMMA':<18}"
    )

    print("-" * 65)

    for token in doc:

        print(
            f"{token.text:<18}"
            f"{token.pos_:<12}"
            f"{token.tag_:<12}"
            f"{token.lemma_:<18}"
        )


# =========================================================
# Dependency Parsing
# =========================================================

def perform_dependency_parsing(text):

    print()
    print("=" * 65)
    print("DEPENDENCY PARSING")
    print("=" * 65)

    print("Text:", text)

    if nlp is None:
        return None

    doc = nlp(text)

    print()

    print(
        f"{'WORD':<18}"
        f"{'POS':<12}"
        f"{'DEP':<15}"
        f"{'HEAD':<18}"
    )

    print("-" * 65)

    for token in doc:

        print(
            f"{token.text:<18}"
            f"{token.pos_:<12}"
            f"{token.dep_:<15}"
            f"{token.head.text:<18}"
        )

    return doc


# =========================================================
# Binary Bag of Words
# =========================================================

def binary_bow(text):

    print()
    print("=" * 65)
    print("BINARY BAG OF WORDS")
    print("=" * 65)

    print("Text:", text)

    if nlp is not None:

        doc = nlp(text)

        vocabulary = sorted(
            {
                token.lemma_.lower()
                for token in doc
                if not token.is_punct
                and not token.is_space
            }
        )

        sentence_words = {
            token.lemma_.lower()
            for token in doc
            if not token.is_punct
            and not token.is_space
        }

    else:

        words = text.lower().split()

        vocabulary = sorted(
            set(words)
        )

        sentence_words = set(words)

    vector = [
        1 if word in sentence_words else 0
        for word in vocabulary
    ]

    print()
    print("Vocabulary:")
    print(vocabulary)

    print()
    print("Binary Vector:")
    print(vector)

    print()

    print(
        f"{'WORD':<25}"
        f"{'VALUE':<10}"
    )

    print("-" * 35)

    for word, value in zip(
        vocabulary,
        vector
    ):

        print(
            f"{word:<25}"
            f"{value:<10}"
        )

    return vocabulary, vector


# =========================================================
# LIGHT INTERACTIVE DEPENDENCY TREE
# =========================================================

def visualize_dependency_tree(
    text,
    output_file="HOC_light_dependency_tree.html"
):

    if nlp is None:
        return

    doc = nlp(text)

    # -----------------------------------------------------
    # Build dependency relationships
    # -----------------------------------------------------

    children = {}

    for token in doc:
        children[token.i] = []

    for token in doc:

        if token.head.i != token.i:

            children[
                token.head.i
            ].append(token.i)

    # -----------------------------------------------------
    # Find root
    # -----------------------------------------------------

    root = None

    for token in doc:

        if token.dep_ == "ROOT":

            root = token.i

            break

    if root is None:

        print(
            "Could not determine dependency root."
        )

        return

    # -----------------------------------------------------
    # Sort children
    # -----------------------------------------------------

    for node in children:

        children[node].sort()

    # -----------------------------------------------------
    # Calculate levels
    # -----------------------------------------------------

    levels = {}

    def assign_levels(
        node,
        level
    ):

        levels[node] = level

        for child in children[node]:

            assign_levels(
                child,
                level + 1
            )

    assign_levels(
        root,
        0
    )

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

        subtree_width[node] = max(
            1,
            total
        )

        return subtree_width[node]

    calculate_width(root)

    # -----------------------------------------------------
    # Canvas dimensions
    # -----------------------------------------------------

    horizontal_spacing = 190
    vertical_spacing = 150

    margin_x = 120
    margin_y = 100

    canvas_width = max(
        1400,
        subtree_width[root]
        * horizontal_spacing
        + margin_x * 2
    )

    max_depth = max(
        levels.values()
    )

    canvas_height = max(
        900,
        (max_depth + 1)
        * vertical_spacing
        + margin_y * 2
    )

    # -----------------------------------------------------
    # Calculate tree positions
    # -----------------------------------------------------

    logical_positions = {}

    def assign_positions(
        node,
        left_boundary
    ):

        if not children[node]:

            logical_positions[node] = (
                left_boundary + 0.5
            )

            return left_boundary + 1

        current_left = left_boundary

        child_positions = []

        for child in children[node]:

            current_left = assign_positions(
                child,
                current_left
            )

            child_positions.append(
                logical_positions[child]
            )

        logical_positions[node] = (
            min(child_positions)
            +
            max(child_positions)
        ) / 2

        return current_left

    assign_positions(
        root,
        0
    )

    # -----------------------------------------------------
    # Convert positions
    # -----------------------------------------------------

    coordinates = {}

    for node, position in logical_positions.items():

        coordinates[node] = {

            "x":
                margin_x
                +
                position
                *
                horizontal_spacing,

            "y":
                margin_y
                +
                levels[node]
                *
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

    for token in doc:

        node = token.i

        x = coordinates[node]["x"]
        y = coordinates[node]["y"]

        node_class = "node"

        if node == root:

            node_class += " root-node"

        node_svg += f"""
        <g
            id="node-{node}"
            class="{node_class}"
            data-node="{node}"
            transform="translate({x},{y})"
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

<html lang="en">

<head>

<meta charset="UTF-8">

<title>HOC Dependency Tree</title>

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

.workspace {{

    width: 100%;
    height: 100%;

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

.edge {{

    stroke: #333;

    stroke-width: 2.5;

    fill: none;

    pointer-events: none;
}}

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

.root-node circle {{

    fill: #e8f0ff;

    stroke: #1d4ed8;

    stroke-width: 4;
}}

.selected circle {{

    stroke: #dc2626;

    stroke-width: 4;
}}

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

</style>

</head>

<body>

<div class="workspace">

<svg
    id="dependencyTree"
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

<g id="edges">

{edge_svg}

</g>

<g id="nodes">

{node_svg}

</g>

</svg>

</div>

<script>

const svg =
    document.getElementById(
        "dependencyTree"
    );


const positions = {{

"""

    for node, position in coordinates.items():

        html += f"""
    {node}: {{
        x: {position['x']},
        y: {position['y']}
    }},
"""


    html += """
};


const originalPositions =
    JSON.parse(
        JSON.stringify(
            positions
        )
    );


const connections = [

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

    return point.matrixTransform(
        matrix
    );
}


function updateEdges() {

    connections.forEach(
        connection => {

            const parent =
                positions[
                    connection.parent
                ];

            const child =
                positions[
                    connection.child
                ];

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

            const startX =
                parent.x +
                (dx / distance) *
                radius;

            const startY =
                parent.y +
                (dy / distance) *
                radius;

            const endX =
                child.x -
                (dx / distance) *
                radius;

            const endY =
                child.y -
                (dy / distance) *
                radius;

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

        }
    );

}


let draggedNode = null;

let offsetX = 0;

let offsetY = 0;


document
    .querySelectorAll(".node")
    .forEach(node => {

        node.addEventListener(
            "pointerdown",
            event => {

                event.preventDefault();

                draggedNode =
                    Number(
                        node.dataset.node
                    );

                const point =
                    getSVGPoint(event);

                offsetX =
                    positions[
                        draggedNode
                    ].x -
                    point.x;

                offsetY =
                    positions[
                        draggedNode
                    ].y -
                    point.y;

                node.setPointerCapture(
                    event.pointerId
                );

                document
                    .querySelectorAll(".node")
                    .forEach(item => {

                        item.classList.remove(
                            "selected"
                        );

                    });

                node.classList.add(
                    "selected"
                );

            }
        );

    });


svg.addEventListener(
    "pointermove",
    event => {

        if (draggedNode === null) {
            return;
        }

        const point =
            getSVGPoint(event);

        let x =
            point.x +
            offsetX;

        let y =
            point.y +
            offsetY;

        const radius = 48;

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

        positions[
            draggedNode
        ].x = x;

        positions[
            draggedNode
        ].y = y;

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


svg.addEventListener(
    "pointerup",
    () => {

        draggedNode = null;

    }
);


svg.addEventListener(
    "pointercancel",
    () => {

        draggedNode = null;

    }
);


svg.addEventListener(
    "dblclick",
    event => {

        if (event.target !== svg) {
            return;
        }

        Object.keys(
            originalPositions
        ).forEach(id => {

            positions[id].x =
                originalPositions[id].x;

            positions[id].y =
                originalPositions[id].y;

            const node =
                document.getElementById(
                    `node-${id}`
                );

            if (node) {

                node.setAttribute(
                    "transform",
                    `translate(
                        ${positions[id].x},
                        ${positions[id].y}
                    )`
                );

            }

        });

        document
            .querySelectorAll(".node")
            .forEach(item => {

                item.classList.remove(
                    "selected"
                );

            });

        updateEdges();

    }
);


updateEdges();

</script>

</body>

</html>
"""


    output = Path(
        output_file
    )

    output.write_text(
        html,
        encoding="utf-8"
    )

    print()
    print(
        "Interactive light dependency tree generated:"
    )

    print(
        output.absolute()
    )

    print()

    webbrowser.open(
        output.absolute().as_uri()
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else
        "The aeroplane fly in the blue sky"
    )

    print()
    print("=" * 65)
    print("HOC - MODIFIED SENTENCE PIPELINE")
    print("=" * 65)

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------

    original_text = target.strip()

    if not original_text:

        print("No input text received.")
        sys.exit(1)

    print()
    print("Original sentence:")
    print(original_text)

    # -----------------------------------------------------
    # Modify word order
    # -----------------------------------------------------

    modified_text = modify_sentence(
        original_text
    )

    print()
    print("Modified sentence:")
    print(modified_text)

    # -----------------------------------------------------
    # POS tagging
    # -----------------------------------------------------

    perform_pos_tagging(
        modified_text
    )

    # -----------------------------------------------------
    # Dependency parsing
    # -----------------------------------------------------

    perform_dependency_parsing(
        modified_text
    )

    # -----------------------------------------------------
    # Visual dependency tree
    # -----------------------------------------------------

    visualize_dependency_tree(
        modified_text
    )

    # -----------------------------------------------------
    # Binary BoW
    # -----------------------------------------------------

    binary_bow(
        modified_text
    )