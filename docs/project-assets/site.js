(() => {
  const header = document.querySelector("[data-header]");
  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.getElementById("navLinks");
  const loadDemoButton = document.querySelector("[data-load-demo]");
  const demoPreview = document.querySelector("[data-demo-preview]");
  const copyButton = document.querySelector("[data-copy-citation]");

  const syncHeader = () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 18);
  };
  syncHeader();
  window.addEventListener("scroll", syncHeader, { passive: true });

  navToggle?.addEventListener("click", () => {
    const open = navToggle.getAttribute("aria-expanded") !== "true";
    navToggle.setAttribute("aria-expanded", String(open));
    navLinks?.classList.toggle("is-open", open);
  });

  navLinks?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      navToggle?.setAttribute("aria-expanded", "false");
      navLinks.classList.remove("is-open");
    });
  });

  const revealItems = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -30px" });
    revealItems.forEach((item) => observer.observe(item));
  } else {
    revealItems.forEach((item) => item.classList.add("is-visible"));
  }

  loadDemoButton?.addEventListener("click", () => {
    if (demoPreview?.classList.contains("is-live")) return;
    const iframe = document.createElement("iframe");
    iframe.src = "physical_layer_dashboardV2.html";
    iframe.title = "Interactive CCS_RL physical layer operations replay";
    iframe.loading = "lazy";
    iframe.setAttribute("allowfullscreen", "");
    demoPreview.replaceChildren(iframe);
    demoPreview.classList.add("is-live");
    loadDemoButton.textContent = "Replay loaded";
    loadDemoButton.disabled = true;
  });

  copyButton?.addEventListener("click", async () => {
    const citation = document.getElementById("citationText")?.textContent || "";
    try {
      await navigator.clipboard.writeText(citation);
      copyButton.textContent = "Copied";
    } catch {
      copyButton.textContent = "Select and copy";
    }
    window.setTimeout(() => {
      copyButton.textContent = "Copy BibTeX";
    }, 1800);
  });
})();
