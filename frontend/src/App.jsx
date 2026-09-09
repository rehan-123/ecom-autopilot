import React from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import {
  Analytics, Approvals, Dashboard, Discovery, Inventory, Ledger,
  Opportunities, Orders, ProductDetail, Settings, Suppliers,
} from './pages'
import Research from './pages/Research'
import Marketplaces from './pages/Marketplaces'
import Execution from './pages/Execution'

const NAV = [
  ['/', 'Dashboard'],
  ['/research', 'AI Research'],
  ['/discovery', 'Discovery'],
  ['/suppliers', 'Suppliers'],
  ['/opportunities', 'Opportunities'],
  ['/approvals', 'Approval queue'],
  ['/marketplaces', 'Marketplaces'],
  ['/execution', 'Execution'],
  ['/orders', 'Orders'],
  ['/inventory', 'Inventory'],
  ['/analytics', 'Analytics'],
  ['/ledger', 'Audit ledger'],
  ['/settings', 'Settings'],
]

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">Ecom Autopilot<small>paper mode · human approval</small></div>
        <nav className="nav">
          {NAV.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === '/'}
              className={({ isActive }) => (isActive ? 'active' : undefined)}>
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/research" element={<Research />} />
          <Route path="/discovery" element={<Discovery />} />
          <Route path="/suppliers" element={<Suppliers />} />
          <Route path="/opportunities" element={<Opportunities />} />
          <Route path="/products/:key" element={<ProductDetail />} />
          <Route path="/approvals" element={<Approvals />} />
          <Route path="/marketplaces" element={<Marketplaces />} />
          <Route path="/execution" element={<Execution />} />
          <Route path="/orders" element={<Orders />} />
          <Route path="/inventory" element={<Inventory />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/ledger" element={<Ledger />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  )
}
