from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment='Attention Gate', format='png')
dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.5')

# Define nodes (operations)
dot.node('g', 'Input (g)\n[from lower level]', shape='box')
dot.node('s', 'Input (s)\n[skip connection]', shape='box')
dot.node('Wg', '1x1 Conv + BN\n(in_c[0]→out_c)', shape='box')
dot.node('Ws', '1x1 Conv + BN\n(in_c[1]→out_c)', shape='box')
dot.node('add', 'Add\n(Wg + Ws)', shape='circle', width='0.6')
dot.node('relu', 'ReLU', shape='box')
dot.node('att', '1x1 Conv + Sigmoid\n(out_c→out_c)', shape='box')
dot.node('multiply', 'Multiply\n(att * s)', shape='circle', width='0.6')
dot.node('out', 'Output\n(weighted features)', shape='box')

# Connect the nodes
dot.edges([
    ('g', 'Wg'),
    ('s', 'Ws'),
    ('Wg', 'add'),
    ('Ws', 'add'),
    ('add', 'relu'),
    ('relu', 'att'),
    ('att', 'multiply'),
    ('s', 'multiply:l'),  # Connect 's' to left side of multiply
    ('multiply', 'out')
])

# Style adjustments
dot.attr('node', style='filled', color='lightgrey', fontname='Helvetica')
dot.attr('edge', arrowsize='0.7')

# Save and render
dot.render('results/attention_gate', view=True, cleanup=True)