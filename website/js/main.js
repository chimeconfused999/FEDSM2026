// Navigation scroll effect
const nav = document.querySelector('.nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 40);
});

// Mobile menu toggle
const toggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');
if (toggle) {
  toggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
  navLinks.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => navLinks.classList.remove('open'));
  });
}

// Scroll reveal
const revealEls = document.querySelectorAll('.reveal');
const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
);
revealEls.forEach(el => observer.observe(el));

// Gallery frame tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const galleryFrames = document.querySelectorAll('[data-frame]');

tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const frame = btn.dataset.frame;
    tabBtns.forEach(b => b.classList.toggle('active', b === btn));
    galleryFrames.forEach(col => {
      col.style.display = col.dataset.frame === frame ? '' : 'none';
    });
  });
});

// Smooth counter animation for hero stats
function animateValue(el, end, suffix = '', decimals = 0) {
  const duration = 1400;
  const start = performance.now();
  const from = 0;
  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    const val = from + (end - from) * eased;
    el.textContent = decimals > 0 ? val.toFixed(decimals) + suffix : Math.round(val) + suffix;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

const statsObs = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const val = parseFloat(el.dataset.value);
      const suffix = el.dataset.suffix || '';
      const decimals = parseInt(el.dataset.decimals || '0', 10);
      animateValue(el, val, suffix, decimals);
      statsObs.unobserve(el);
    });
  },
  { threshold: 0.5 }
);

document.querySelectorAll('[data-value]').forEach(el => statsObs.observe(el));
