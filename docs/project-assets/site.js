(() => {
  const header = document.querySelector("[data-header]");
  const navToggle = document.querySelector(".nav-toggle");
  const navLinks = document.getElementById("navLinks");
  const loadDemoButton = document.querySelector("[data-load-demo]");
  const demoPreview = document.querySelector("[data-demo-preview]");
  const copyButton = document.querySelector("[data-copy-citation]");
  const lightThemeStyles = document.getElementById("lightThemeStyles");
  const themeChoices = document.querySelectorAll("[data-theme-choice]");
  const themeColorMeta = document.querySelector('meta[name="theme-color"]');

  const applyTheme = (theme, persist = true) => {
    const normalized = theme === "dark" ? "dark" : "light-v2";
    document.documentElement.dataset.theme = normalized;
    if (lightThemeStyles) {
      lightThemeStyles.media = normalized === "dark" ? "not all" : "all";
    }
    themeChoices.forEach((button) => {
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.themeChoice === normalized),
      );
    });
    themeColorMeta?.setAttribute(
      "content",
      normalized === "dark" ? "#030b11" : "#f6f9f7",
    );
    demoPreview
      ?.querySelector("iframe")
      ?.contentWindow
      ?.postMessage({ type: "ccs-rl-theme", theme: normalized }, "*");
    if (persist) {
      try {
        localStorage.setItem("ccs-rl-dashboard-theme", normalized);
      } catch {
        // The selected theme still applies for the current page.
      }
    }
  };

  themeChoices.forEach((button) => {
    button.addEventListener("click", () => {
      applyTheme(button.dataset.themeChoice);
    });
  });
  applyTheme(document.documentElement.dataset.theme, false);
  window.addEventListener("storage", (event) => {
    if (
      event.key === "ccs-rl-dashboard-theme"
      && (event.newValue === "light-v2" || event.newValue === "dark")
    ) {
      applyTheme(event.newValue, false);
    }
  });

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
