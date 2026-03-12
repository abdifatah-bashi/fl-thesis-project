import defaultComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { Step, Steps } from 'fumadocs-ui/components/steps';
import { Tab, Tabs } from 'fumadocs-ui/components/tabs';
import { Mermaid } from '@/components/mermaid';

export function getMDXComponents(components: MDXComponents): MDXComponents {
  return {
    ...defaultComponents,
    Step,
    Steps,
    Tab,
    Tabs,
    Mermaid,
    ...components,
  };
}
