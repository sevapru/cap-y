/**
 * wheels.sobaka.dev -- PEP 503 Simple Repository API for Jetson Thor wheels
 *
 * Serves pre-built Python wheels from Cloudflare R2:
 *   - OpenCV 4.13 CUDA (sm_110, cuDNN, cuBLAS, FAST_MATH)
 *   - Open3D 0.19+ CUDA (PyTorch ops, RealSense, GUI)
 *   - JAX/jaxlib CUDA 13 (SM 110 native kernels)
 *
 * Usage:
 *   uv pip install --extra-index-url https://wheels.sobaka.dev/simple/ jaxlib
 *
 * R2 bucket structure:
 *   whl/opencv_python_headless-4.13.0-cp312-...aarch64.whl
 *   whl/open3d-0.19.0-cp312-...aarch64.whl
 *   whl/jaxlib-0.9.2-cp312-...aarch64.whl
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    const headers = {
      'Access-Control-Allow-Origin': '*',
      'Cache-Control': 'public, max-age=3600',
    };

    // Root: landing page
    if (path === '/' || path === '') {
      return new Response(landingPage(), {
        headers: { ...headers, 'Content-Type': 'text/html' },
      });
    }

    // PEP 503: /simple/ -- list all packages
    if (path === '/simple/' || path === '/simple') {
      const packages = await listPackages(env.BUCKET);
      const html = simpleIndex(packages);
      return new Response(html, {
        headers: { ...headers, 'Content-Type': 'text/html' },
      });
    }

    // PEP 503: /simple/<package>/ -- list wheels for a package
    const pkgMatch = path.match(/^\/simple\/([^/]+)\/?$/);
    if (pkgMatch) {
      const pkgName = pkgMatch[1];
      const wheels = await listWheels(env.BUCKET, pkgName);
      if (wheels.length === 0) {
        return new Response('Not Found', { status: 404 });
      }
      const html = packageIndex(pkgName, wheels);
      return new Response(html, {
        headers: { ...headers, 'Content-Type': 'text/html' },
      });
    }

    // Direct wheel download: /whl/<filename>.whl
    const whlMatch = path.match(/^\/whl\/(.+\.whl)$/);
    if (whlMatch) {
      const key = `whl/${whlMatch[1]}`;
      const object = await env.BUCKET.get(key);
      if (!object) {
        return new Response('Not Found', { status: 404 });
      }
      return new Response(object.body, {
        headers: {
          ...headers,
          'Content-Type': 'application/octet-stream',
          'Content-Disposition': `attachment; filename="${whlMatch[1]}"`,
          'Content-Length': object.size,
        },
      });
    }

    return new Response('Not Found', { status: 404 });
  },
};

async function listPackages(bucket) {
  const listed = await bucket.list({ prefix: 'whl/', delimiter: '/' });
  const packages = new Set();

  for (const obj of listed.objects || []) {
    const filename = obj.key.replace('whl/', '');
    const pkgName = filename.split('-')[0].toLowerCase().replace(/_/g, '-');
    packages.add(pkgName);
  }

  return [...packages].sort();
}

async function listWheels(bucket, pkgName) {
  const listed = await bucket.list({ prefix: 'whl/' });
  const normalized = pkgName.toLowerCase().replace(/-/g, '_');

  return (listed.objects || [])
    .map((obj) => obj.key.replace('whl/', ''))
    .filter((name) => {
      const whlPkg = name.split('-')[0].toLowerCase();
      return whlPkg === normalized || whlPkg === pkgName.toLowerCase();
    });
}

function simpleIndex(packages) {
  const links = packages.map((p) => `<a href="/simple/${p}/">${p}</a>`).join('\n');
  return `<!DOCTYPE html>
<html><head><title>Jetson Thor Wheels</title></head>
<body><h1>Jetson Thor Wheels (sobaka.dev)</h1>
${links}
</body></html>`;
}

function packageIndex(pkgName, wheels) {
  const links = wheels
    .map((w) => `<a href="/whl/${w}">${w}</a>`)
    .join('\n');
  return `<!DOCTYPE html>
<html><head><title>${pkgName}</title></head>
<body><h1>${pkgName}</h1>
${links}
</body></html>`;
}

function landingPage() {
  return `<!DOCTYPE html>
<html><head><title>wheels.sobaka.dev</title></head>
<body>
<h1>Jetson Thor Python Wheels</h1>
<p>Pre-built CUDA-accelerated wheels for NVIDIA Jetson Thor (aarch64, SM 110, CUDA 13.0).</p>
<h2>Usage</h2>
<pre>uv pip install --extra-index-url https://wheels.sobaka.dev/simple/ jaxlib open3d opencv-python-headless</pre>
<h2>Available packages</h2>
<ul>
<li><strong>opencv-python-headless</strong> 4.13 -- CUDA DNN, cuBLAS, cuDNN 9.12, FAST_MATH, GStreamer, NEON</li>
<li><strong>open3d</strong> 0.19+ -- CUDA, PyTorch ops, RealSense, Open3D-ML, GUI</li>
<li><strong>jaxlib</strong> + <strong>jax-cuda13-plugin</strong> -- SM 110 native kernels (no PTX JIT)</li>
</ul>
<p><a href="/simple/">PEP 503 Simple Index</a></p>
<p>Source: <a href="https://github.com/sevapru/cap-y">github.com/sevapru/cap-y</a></p>
</body></html>`;
}
