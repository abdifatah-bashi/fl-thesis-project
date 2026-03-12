import type { Plugin } from 'unified';
import type { Root, Code } from 'mdast';
import { visit } from 'unist-util-visit';

/**
 * Remark plugin that converts ```mermaid code blocks
 * into <Mermaid chart="..." /> JSX elements.
 */
const remarkMermaid: Plugin<[], Root> = () => {
  return (tree) => {
    visit(tree, 'code', (node: Code, index, parent) => {
      if (node.lang !== 'mermaid' || !parent || index === undefined) return;

      // Replace the code block with an MDX JSX element
      parent.children[index] = {
        type: 'mdxJsxFlowElement' as any,
        name: 'Mermaid',
        attributes: [
          {
            type: 'mdxJsxAttribute' as any,
            name: 'chart',
            value: node.value,
          },
        ],
        children: [],
      } as any;
    });
  };
};

export default remarkMermaid;
