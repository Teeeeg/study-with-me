// Derives the cluster (the folder a note lives in) and keeps unfinished,
// front-matter-less placeholder notes out of the build.
export default {
  cluster: (data) => {
    const stem = data.page?.filePathStem ?? "";
    return stem.slice(0, stem.lastIndexOf("/") + 1);
  },

  eleventyExcludeFromCollections: (data) =>
    data.eleventyExcludeFromCollections === true || !data.ref,

  permalink: (data) => {
    if (!data.ref) return false;

    const stem = data.page.filePathStem.replace(/\.zh-CN$/, "");
    const base = stem === "/index" ? "" : stem;
    const prefix = data.lang === "zh" ? "/zh" : "";
    return `${prefix}${base}/index.html`;
  },
};
