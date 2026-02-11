/**
 * FL Framework Docs — Premium Interactions
 */

document.addEventListener('DOMContentLoaded', () => {

  /* ---- Intersection Observer: Fade-in on Scroll ---- */
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  );

  document.querySelectorAll('.animate-on-scroll, .fl-card, .fl-stat').forEach((el) => {
    observer.observe(el);
  });


  /* ---- Animated Counters ---- */
  const counters = document.querySelectorAll('.fl-stat-number[data-target]');
  const counterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const el = entry.target;
          const target = parseInt(el.getAttribute('data-target'), 10);
          const suffix = el.getAttribute('data-suffix') || '';
          animateCounter(el, target, suffix);
          counterObserver.unobserve(el);
        }
      });
    },
    { threshold: 0.5 }
  );

  counters.forEach((c) => counterObserver.observe(c));

  function animateCounter(el, target, suffix) {
    const duration = 1200;
    const start = performance.now();

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const current = Math.round(eased * target);
      el.textContent = current.toLocaleString() + suffix;

      if (progress < 1) {
        requestAnimationFrame(tick);
      }
    }
    requestAnimationFrame(tick);
  }

});


/* ---- Re-observe after MkDocs instant navigation ---- */
if (typeof document$ !== 'undefined') {
  document$.subscribe(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
    );

    document.querySelectorAll('.animate-on-scroll, .fl-card, .fl-stat').forEach((el) => {
      el.classList.remove('visible');
      observer.observe(el);
    });

    /* Re-run counters */
    const counterObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const el = entry.target;
            const target = parseInt(el.getAttribute('data-target'), 10);
            const suffix = el.getAttribute('data-suffix') || '';
            const duration = 1200;
            const start = performance.now();
            function tick(now) {
              const elapsed = now - start;
              const progress = Math.min(elapsed / duration, 1);
              const eased = 1 - Math.pow(1 - progress, 3);
              el.textContent = Math.round(eased * target).toLocaleString() + suffix;
              if (progress < 1) requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
            counterObserver.unobserve(el);
          }
        });
      },
      { threshold: 0.5 }
    );
    document.querySelectorAll('.fl-stat-number[data-target]').forEach((c) => counterObserver.observe(c));
  });
}
