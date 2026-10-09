export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#F3F4F6', surface: '#FFFFFF', ink: '#111827', muted: '#6B7280', line: '#E5E7EB',
        teal: { DEFAULT: '#0F766E', dk: '#115E59', lt: '#CCFBF1' },
        urgent: '#DC2626', review: '#D97706', routine: '#059669',
      },
      fontFamily: {
        display: ['Outfit', 'Inter', 'system-ui', 'sans-serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: { 
        card: '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
        premium: '0 10px 40px -10px rgba(15, 118, 110, 0.15)'
      },
      borderRadius: { xl2: '20px' },
      backgroundImage: {
        'gradient-premium': 'linear-gradient(135deg, #0F766E 0%, #115E59 100%)',
      }
    },
  },
  plugins: [],
};
