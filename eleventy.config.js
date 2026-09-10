import path from "node:path";
import yaml from "js-yaml";
import syntaxHighlight from "@11ty/eleventy-plugin-syntaxhighlight";

// Change this if the repository is renamed, or set it to "/" for a user site.
const PATH_PREFIX = process.env.PATH_PREFIX ?? "/study-with-me/";

const leafTitle = (dir) => {
  const leaf = dir.split("/").filter(Boolean).pop() ?? dir;
  const words = leaf.replace(/-/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
};

export default function (eleventyConfig) {
  eleventyConfig.addPlugin(syntaxHighlight);

  eleventyConfig.addDataExtension("yml,yaml", (contents) =>
    yaml.load(contents),
  );

  eleventyConfig.addPassthroughCopy("assets");
  eleventyConfig.addPassthroughCopy(".asset");

  // Data-cascade defaults. Front matter overrides all of these.
  eleventyConfig.addGlobalData("layout", "doc.njk");
  eleventyConfig.addGlobalData("lang", "en");
  eleventyConfig.addGlobalData("nav_order", 100);

  eleventyConfig.addCollection("docs", (api) =>
    api
      .getAll()
      .filter((item) => item.data.ref && item.data.nav !== false)
      .sort((a, b) => (a.data.nav_order ?? 100) - (b.data.nav_order ?? 100)),
  );

  eleventyConfig.addFilter("byRefLang", (items, ref, lang) =>
    (items ?? []).find(
      (item) => item.data.ref === ref && item.data.lang === lang,
    ),
  );

  /**
   * Clusters listed in _data/clusters.yml come first, in file order. Any folder
   * that is not listed is appended afterwards and titled from its folder name,
   * so a new section appears without touching the config.
   */
  eleventyConfig.addFilter("clusterGroups", (docs, lang, clusters = []) => {
    const inLang = (docs ?? []).filter((doc) => doc.data.lang === lang);
    const groups = [];
    const listed = new Set();

    for (const cluster of clusters) {
      const items = inLang.filter((doc) => doc.data.cluster === cluster.dir);
      if (!items.length) continue;
      listed.add(cluster.dir);
      groups.push({
        dir: cluster.dir,
        icon: cluster.icon,
        title: cluster.title?.[lang] ?? leafTitle(cluster.dir),
        summary: cluster.summary?.[lang],
        items,
      });
    }

    const extras = new Map();
    for (const doc of inLang) {
      const dir = doc.data.cluster;
      if (listed.has(dir)) continue;
      if (!extras.has(dir)) extras.set(dir, []);
      extras.get(dir).push(doc);
    }

    for (const [dir, items] of [...extras].sort((a, b) =>
      a[0].localeCompare(b[0]),
    )) {
      groups.push({ dir, title: leafTitle(dir), items });
    }

    return groups;
  });

  eleventyConfig.addFilter(
    "backlogGroups",
    (backlog, lang, clusters = [], all = []) =>
      (backlog ?? []).map((group) => {
        let title;
        if (group.dir) {
          const cluster = clusters.find((c) => c.dir === group.dir);
          title = cluster?.title?.[lang] ?? leafTitle(group.dir);
        } else {
          title = group.title?.[lang];
        }

        return {
          title,
          items: (group.items ?? []).map((item) => ({
            status: item.status ?? "todo",
            text: item[lang],
            note: item.note?.[lang],
            page: item.ref
              ? all.find((p) => p.data.ref === item.ref && p.data.lang === lang)
              : null,
          })),
        };
      }),
  );

  eleventyConfig.addFilter("backlogStats", (backlog) => {
    const items = (backlog ?? []).flatMap((group) => group.items ?? []);
    return {
      total: items.length,
      done: items.filter((item) => item.status === "done").length,
    };
  });

  /**
   * Notes link to shared images with paths like `../../.asset/x.svg` so they
   * also render on GitHub. Eleventy emits each note as its own directory, which
   * adds a path segment, so rewrite image sources against the source folder.
   */
  eleventyConfig.addTransform("resolve-relative-images", function (content) {
    if (!this.page.outputPath || !this.page.outputPath.endsWith(".html"))
      return content;

    const sourceDir = path.posix.dirname(
      this.page.inputPath.replace(/^\.\//, ""),
    );
    return content.replace(/src="((?:\.\.\/)+[^"]*)"/g, (match, relative) => {
      const resolved = path.posix.normalize(
        path.posix.join(sourceDir, relative),
      );
      return `src="${path.posix.join(PATH_PREFIX, resolved)}"`;
    });
  });

  return {
    pathPrefix: PATH_PREFIX,
    dir: {
      input: ".",
      output: "_site",
      includes: "_includes",
      layouts: "_layouts",
      data: "_data",
    },
    // Notes are plain Markdown; do not run them through a template engine.
    markdownTemplateEngine: false,
    htmlTemplateEngine: "njk",
  };
}
